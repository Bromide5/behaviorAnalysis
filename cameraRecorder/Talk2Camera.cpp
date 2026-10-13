#include "Talk2Camera.h"

#include <chrono>
#include <ctime>
#include <direct.h>   // _mkdir
#include <io.h>       // _access

using namespace Pylon;
using namespace std;
using namespace Basler_UsbCameraParams;
using namespace cv;

// 逐级创建目录。_mkdir 只能建一级，这里按分隔符逐段创建，
// 支持 "a/b/c/" 以及 "D:/data/trial1/" 这类路径，已存在的层级自动跳过。
static void createDirectories(string dir)
{
    while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\'))
        dir.pop_back();               // 去掉末尾分隔符，避免把空段当目录

    for (size_t i = 1; i <= dir.size(); ++i)
    {
        bool atEnd = (i == dir.size());
        if (!atEnd && dir[i] != '/' && dir[i] != '\\')
            continue;

        string part = dir.substr(0, i);
        // 跳过盘符（"C:"）和根目录（"/"、"/"）片段
        if (part.back() == ':')
            continue;
        if (part.size() == 1 && (part[0] == '/' || part[0] == '\\'))
            continue;

        if (_access(part.c_str(), 0) != 0)
            _mkdir(part.c_str());
    }
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------

bool PylonCameraData::initialize()
{
    const float fps = static_cast<float>(cfg.frameRate);

    if (!initialize_Camera(fps))
        return false;

    initialize_WriteOut(cfg.path, cfg.baseName, fps);
    return true;
}

bool PylonCameraData::initialize_Camera(float frameRate_)
{
    PylonInitialize();
    info.SetDeviceClass(CBaslerUsbInstantCamera::DeviceClass());

    CTlFactory& TlFactory(CTlFactory::GetInstance());
    try
    {
        camera.Attach(TlFactory.CreateFirstDevice(info));
        camera.Open();
    }
    catch (const GenericException& e)
    {
        cerr << "[相机] 打开失败: " << e.GetDescription() << endl;
        return false;
    }

    // 相机名（便于确认连的是哪台）
    try
    {
        cout << "[相机] 已连接: " << camera.GetDeviceInfo().GetModelName()
             << " (" << camera.GetDeviceInfo().GetSerialNumber() << ")" << endl;
    }
    catch (...) {}

    // 从 Pylon Viewer 导出的参数文件加载配置。文件不存在时静默回退到配置项/内置默认值。
    const string pfs = cfg.pfsFile;
    if (!pfs.empty() && _access(pfs.c_str(), 0) == 0)
    {
        try
        {
            CFeaturePersistence::Load(pfs.c_str(), &camera.GetNodeMap(), true);
            cout << "[相机] 已从 " << pfs << " 载入参数" << endl;
        }
        catch (const GenICam::GenericException& e)
        {
            cerr << "[相机] 载入 " << pfs << " 失败，改用配置项: "
                 << e.GetDescription() << endl;
        }
    }
    else if (!pfs.empty())
    {
        cout << "[相机] 未找到 " << pfs << "，改用配置项" << endl;
    }

    // 逐项设参数。个别参数在某些机型上可能不可写，GenApi 会抛异常；
    // 兜住异常，不让单个参数导致整个程序起不来。
    auto setInt = [&](auto& node, int value, const char* name) {
        try {
            node.SetValue(value);
        } catch (const GenericException& e) {
            cerr << "[相机] 设置 " << name << " 失败: " << e.GetDescription() << endl;
        }
    };

    camera.MaxNumBuffer = 1;
    setInt(camera.Width,  CCD_WIDTH,  "Width");
    setInt(camera.Height, CCD_HEIGHT, "Height");
    try { camera.AcquisitionMode.SetValue(AcquisitionMode_Continuous); }
    catch (const GenericException& e) { cerr << "[相机] 设置 AcquisitionMode 失败: " << e.GetDescription() << endl; }
    try { camera.DeviceLinkThroughputLimit.SetValue(
              static_cast<int64_t>(cfg.deviceLinkThroughput)); }
    catch (const GenericException& e) {
        cerr << "[相机] 设置 DeviceLinkThroughputLimit 失败: " << e.GetDescription() << endl;
    }

    if (hardwareTrigger)
    {
        cout << "[相机] 触发模式: 硬件触发 (" << cfg.triggerSource << " 上升沿)" << endl;
        try {
            camera.AcquisitionFrameRateEnable.SetValue(false);
            camera.TriggerSelector.SetIntValue(TriggerSelector_FrameStart);
            camera.TriggerMode.SetValue(TriggerMode_On);
            // 触发源由配置指定（线名，如 Line3 / Line4）。TriggerSource 是枚举节点，
            // 用 GenApi 枚举接口按字符串设置，避免写死成某个 TriggerSource_ 常量。
            GenApi::CEnumerationPtr ts(camera.GetNodeMap().GetNode("TriggerSource"));
            if (ts.IsValid())
                ts->FromString(cfg.triggerSource.c_str());
            else
                cerr << "[相机] 找不到 TriggerSource 节点，触发源未设置" << endl;
            camera.TriggerActivation.SetValue(TriggerActivation_RisingEdge);
        } catch (const GenericException& e) {
            cerr << "[相机] 配置硬件触发失败: " << e.GetDescription() << endl;
        }
    }
    else
    {
        cout << "[相机] 触发模式: 自由运行, " << frameRate_ << " fps" << endl;
        try {
            camera.AcquisitionFrameRateEnable.SetValue(true);
            camera.AcquisitionFrameRate.SetValue(frameRate_);
        } catch (const GenericException& e) {
            cerr << "[相机] 设置帧率失败: " << e.GetDescription() << endl;
        }
    }

    setInt(camera.ExposureTime, exposureTime, "ExposureTime");
    setInt(camera.OffsetY, offsetY, "OffsetY");
    setInt(camera.OffsetX, offsetX, "OffsetX");

    // 输出统一为 Mono8 灰度
    formatConverter.OutputPixelFormat = PixelType_Mono8;

    try
    {
        camera.StartGrabbing(GrabStrategy_LatestImageOnly);
    }
    catch (const GenericException& e)
    {
        cerr << "[相机] StartGrabbing 失败: " << e.GetDescription() << endl;
        return false;
    }

    return true;
}

void PylonCameraData::initialize_WriteOut(string path, string baseName, float frameRate_)
{
    folderName = path;
    if (!folderName.empty())
    {
        char last = folderName.back();
        if (last != '/' && last != '\\')
            folderName += '/';
    }
    folderName += baseName + "/";

    if (_access(folderName.c_str(), 0) != 0)
        createDirectories(folderName);

    // 编码 fourcc 取自成员 codec（命令行可设）。容器用 avi，兼容 XVID/MJPG/FFV1。
    string cc = codec;
    if (cc.size() < 4)
        cc.resize(4, ' ');            // 不足 4 字符右侧补空格
    int fourcc = VideoWriter::fourcc(cc[0], cc[1], cc[2], cc[3]);

    string videoname = folderName + baseName + "_" + getTime() + ".avi";
    video = VideoWriter(videoname, fourcc, frameRate_,
                        Size(CCD_WIDTH, CCD_HEIGHT), /*isColor=*/false);

    if (!video.isOpened())
    {
        cerr << "[录像] 无法用编码 " << codec << " 创建输出文件: " << videoname << endl;
        cerr << "       该编码可能不被 OpenCV/FFmpeg 支持，请换一种（见 README）。" << endl;
        return;
    }

    cout << "[录像] 输出 -> " << videoname << "  (编码 " << codec << ")" << endl;
}

// ---------------------------------------------------------------------------
// 取帧
// ---------------------------------------------------------------------------

bool PylonCameraData::grabImg()
{
    try
    {
        camera.RetrieveResult(1000, ptrGrabResult, TimeoutHandling_ThrowException);

        if (!ptrGrabResult->GrabSucceeded())
        {
            cerr << "[取帧] 失败: " << ptrGrabResult->GetErrorCode()
                 << " " << ptrGrabResult->GetErrorDescription() << endl;
            return false;
        }

        formatConverter.Convert(pylonImg, ptrGrabResult);

        // 独占写入共享帧。
        //
        // 注意：这里必须 clone()。pylonImg 内部缓冲会在下一次 Convert 时被覆盖，
        // 若不 clone 就等于让所有消费方去读一块随时会被改写的内存。
        // （拾取到的 buffer 属于 grab result，本步结束后即被回收，不能直接长期引用。）
        Mat frame(ptrGrabResult->GetHeight(),
                  ptrGrabResult->GetWidth(),
                  CV_8UC1,
                  (uint8_t*)pylonImg.GetBuffer());

        g_shared_mutex.lock();
        opencvImg = frame.clone();
        g_shared_mutex.unlock();
    }
    catch (const GenericException& e)
    {
        cerr << "[取帧] 异常: " << e.GetDescription() << endl;
        return false;
    }

    return camera.IsGrabbing();
}

void PylonCameraData::captureImg()
{
    cout << "[取帧] 线程启动" << endl;
    while (isCapture.load())
    {
        grabImg();
    }
    cout << "[取帧] 线程退出" << endl;
}

// ---------------------------------------------------------------------------
// 消费方
// ---------------------------------------------------------------------------

void PylonCameraData::writeOutData()
{
    if (!video.isOpened())
        return;

    try
    {
        // 录像只读共享帧，用共享锁。
        g_shared_mutex.lock_shared();
        if (!opencvImg.empty())
        {
            if (opencvImg.cols == CCD_WIDTH && opencvImg.rows == CCD_HEIGHT)
                video.write(opencvImg);
            else
                cerr << "[录像] 帧尺寸异常，跳过: " << opencvImg.cols << "x" << opencvImg.rows << endl;
        }
        g_shared_mutex.unlock_shared();
    }
    catch (const cv::Exception& e)
    {
        cerr << "[录像] 异常: " << e.what() << endl;
    }
}

void PylonCameraData::imgShow()
{
    if (!showWindow)
        return;

    try
    {
        Mat frame;
        g_shared_mutex.lock_shared();
        if (!opencvImg.empty())
            frame = opencvImg.clone();   // 持锁期间只做廉价拷贝
        g_shared_mutex.unlock_shared();

        if (frame.empty())
            return;

        // 中心十字参考线
        line(frame, Point(CCD_WIDTH / 2, 0), Point(CCD_WIDTH / 2, CCD_HEIGHT - 1),
             Scalar(0, 255, 0), 1);
        line(frame, Point(0, CCD_HEIGHT / 2), Point(CCD_WIDTH - 1, CCD_HEIGHT / 2),
             Scalar(0, 255, 0), 1);
        // 横向三等分参考线（与 SPI 分析的缺省 ROI 划分一致）
        for (int i = 1; i <= 2; ++i)
        {
            int x = i * CCD_WIDTH / 3;
            line(frame, Point(x, 0), Point(x, CCD_HEIGHT - 1), Scalar(0, 128, 255), 1);
        }

        Mat show;
        resize(frame, show, Size(CCD_WIDTH / 2, CCD_HEIGHT / 2));

        static bool windowReady = false;
        if (!windowReady)
        {
            moveWindow("image", -23, -3);   // 避免遮挡实验区域
            windowReady = true;
        }
        imshow("image", show);
        waitKey(1);
    }
    catch (const cv::Exception& e)
    {
        cerr << "[显示] 异常: " << e.what() << endl;
    }
}

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

string PylonCameraData::getTime()
{
    time_t nowtime = time(nullptr);
    tm local;
    localtime_s(&local, &nowtime);

    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &local);
    return string(buf);
}
