#include "fishAnalysis.h"

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

#include "lock.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

bool processFrame(const Mat& frame)
{
    // TODO: 在此接入实际的推理 / 分析算法，例如：
    //   - 把 frame 送进 TensorRT 引擎做前向
    //   - 或做背景差分 + 连通域求鱼体位置
    //   - 把结果写入某个结果队列，交给写盘线程
    //
    // 当前占位实现只做最基本的合法性检查。
    if (frame.empty())
        return false;

    // 例如可以在这里算个全图均值：
    // Scalar mean = cv::mean(frame);

    return true;
}

void runInferenceLoop(PylonCameraData& cam, const atomic<bool>& running,
                      double targetFps)
{
    cout << "[推理] 线程启动 (目标 " << targetFps << " fps)" << endl;

    const auto period = microseconds(static_cast<long long>(1e6 / targetFps));
    auto next = steady_clock::now();

    long long processed = 0;
    auto tLastReport = steady_clock::now();

    while (running.load())
    {
        Mat frame;
        {
            // 只读共享帧，持锁期间做一次 clone，随即放锁。
            std::shared_lock<std::shared_mutex> lk(g_shared_mutex);
            if (!cam.opencvImg.empty())
                frame = cam.opencvImg.clone();
        }

        if (!frame.empty() && processFrame(frame))
            ++processed;

        // 每秒汇报一次处理帧率
        auto now = steady_clock::now();
        if (now - tLastReport >= seconds(1))
        {
            double secs = duration<double>(now - tLastReport).count();
            cout << "[推理] " << static_cast<int>(processed / secs) << " fps" << endl;
            processed = 0;
            tLastReport = now;
        }

        // 按目标帧率节拍，避免空转吃满 CPU
        next += period;
        std::this_thread::sleep_until(next);
        if (steady_clock::now() > next + period * 5)   // 落后太多则重新对齐
            next = steady_clock::now();
    }

    cout << "[推理] 线程退出" << endl;
}
