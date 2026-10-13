// 相机录像程序：Basler Pylon + OpenCV，多线程共享帧
//
// 架构（详见 README）：
//   取帧线程  --RetrieveResult-->  共享 cv::Mat opencvImg  <--lock_shared--  录像 / 显示 / 推理
//
// 用法（不接受命令行参数）：
//   直接双击或运行 cameraRecorder.exe。所有可调参数集中在与 exe 同目录的
//   camera_config.txt（键 = 值）。该文件不存在时用内置默认值并自动生成一份。
//   按回车结束录制。

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <windows.h>   // SetConsoleOutputCP
#include <pylon/PylonIncludes.h>

#include "Config.h"
#include "Talk2Camera.h"
#include "fishAnalysis.h"

using namespace std;

// 配置文件缺省名（与 exe 同目录）
static const char* kConfigFile = "camera_config.txt";

// 录像线程：按帧率节拍，把共享帧写进 avi
static void recordLoop(PylonCameraData& cam, const atomic<bool>& running,
                       double fps)
{
    cout << "[录像] 线程启动 (" << fps << " fps)" << endl;
    const auto period = chrono::microseconds(static_cast<long long>(1e6 / fps));
    auto next = chrono::steady_clock::now();

    while (running.load())
    {
        cam.writeOutData();

        next += period;
        this_thread::sleep_until(next);
        if (chrono::steady_clock::now() > next + period * 5)
            next = chrono::steady_clock::now();
    }
    cout << "[录像] 线程退出" << endl;
}

// 显示线程：按帧率节拍刷新窗口
static void showLoop(PylonCameraData& cam, const atomic<bool>& running,
                     double fps)
{
    cout << "[显示] 线程启动" << endl;
    const auto period = chrono::microseconds(static_cast<long long>(1e6 / fps));
    auto next = chrono::steady_clock::now();

    while (running.load())
    {
        cam.imgShow();

        next += period;
        this_thread::sleep_until(next);
        if (chrono::steady_clock::now() > next + period * 5)
            next = chrono::steady_clock::now();
    }
    cout << "[显示] 线程退出" << endl;
}

// 相机生命周期封装在独立作用域里，确保 cam（内含 CBaslerUsbInstantCamera）
// 在 PylonTerminate() 之前就析构完毕。
//
// 若顺序颠倒——先 PylonTerminate() 再让相机对象析构——相机析构会访问已被 pylon
// 回收的内部状态，实测在退出时触发段错误（exit code 139）。
static int runCamera(const Config& cfg)
{
    PylonCameraData cam(cfg);
    if (!cam.initialize())
    {
        cerr << "[错误] 相机初始化失败，退出。" << endl;
        return -1;
    }

    atomic<bool> running{ true };
    cam.isCapture = true;

    // 线程：取帧、录像、推理，以及（可选）显示
    const double recFps = cfg.recordFpsEff();
    const bool   show   = cfg.showWindow;

    thread tGrab(&PylonCameraData::captureImg, &cam);
    thread tRec(recordLoop, std::ref(cam), std::cref(running), recFps);
    thread tShow;
    if (show)
        tShow = thread(showLoop, std::ref(cam), std::cref(running),
                       static_cast<double>(cfg.frameRate));
    thread tInfer(runInferenceLoop, std::ref(cam), std::cref(running),
                  static_cast<double>(cfg.frameRate));

    cout << "\n>>> 录制中。按回车停止...\n" << endl;
    cin.get();

    // 收尾：先停取帧，再停消费方，最后 join
    cout << "\n[主] 停止录制..." << endl;
    cam.isCapture = false;
    running = false;

    tGrab.join();
    tRec.join();
    if (tShow.joinable())
        tShow.join();
    tInfer.join();

    if (cam.video.isOpened())
        cam.video.release();

    return 0;   // cam 在此作用域末尾析构，早于下面的 PylonTerminate()
}

int main(int /*argc*/, char** /*argv*/)
{
    // 源码与字符串都是 UTF-8，把控制台也切到 UTF-8 才能正确打印中文
    SetConsoleOutputCP(CP_UTF8);

    cout << "===== 相机录像程序 =====" << endl;

    // 载入配置。文件不存在则用内置默认值，并生成一份模板。
    Config cfg;
    if (loadConfig(cfg, kConfigFile))
    {
        cout << "[配置] 已载入 " << kConfigFile << endl;
    }
    else
    {
        cout << "[配置] 未找到 " << kConfigFile << "，使用内置默认值" << endl;
        if (writeDefaultConfig(cfg, kConfigFile))
            cout << "[配置] 已生成配置模板，改完重跑即可生效: " << kConfigFile << endl;
        else
            cerr << "[配置] 生成模板失败（目录不可写？），本次仍按默认值运行" << endl;
    }
    printConfig(cfg);

    if (cfg.frameRate <= 0.0)
    {
        cerr << "[错误] 配置中的 frameRate 必须为正数，当前为 "
             << cfg.frameRate << endl;
        Pylon::PylonTerminate();
        return 1;
    }

    const int rc = runCamera(cfg);

    Pylon::PylonTerminate();
    cout << "[主] 已退出。" << endl;
    return rc;
}
