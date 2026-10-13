#pragma once
#ifndef _TALK2CAMERA_H
#define _TALK2CAMERA_H

// pylon libs
#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>
// opencv libs
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// standard c++ libs
#include <atomic>
#include <iostream>
#include <string>

#include "lock.h"
#include "Config.h"

// Basler acA2000-165umNIR 的全画幅尺寸
#define CCD_WIDTH  2048
#define CCD_HEIGHT 1088

class PylonCameraData
{
public:
    // 所有可调参数来自 cfg（由 camera_config.txt 载入，见 Config.h）。
    explicit PylonCameraData(const Config& c) : cfg(c)
    {
        exposureTime = cfg.exposureTime;      // us
        frameRate = static_cast<float>(cfg.frameRate);
        hardwareTrigger = cfg.hardwareTrigger; // false = 自由运行；true = 硬件触发
        offsetY = cfg.offsetY;
        offsetX = cfg.offsetX;
        frameNum = 0;
        imgNum = 0;
        isCapture = true;
        showWindow = cfg.showWindow;
        codec = cfg.codec;                     // 录像编码，见 README「视频编码与体积」
    }

    // ---- 生命周期 ----
    // 打开相机并准备录像输出。任一环节失败返回 false。
    bool initialize();
    bool initialize_Camera(float frameRate);
    void initialize_WriteOut(std::string path, std::string baseName, float frameRate);

    // ---- 取帧 ----
    // 从相机取一帧，转成 Mono8 灰度并写入共享的 opencvImg（独占锁内 clone）。
    // 返回 false 表示本帧失败或相机已停止。
    bool grabImg();

    // 取帧线程主体：只要 isCapture 为真就一直取帧。
    void captureImg();

    // ---- 消费方（各自在共享锁保护下取用 opencvImg）----
    void writeOutData();   // 录像：把当前帧写进 video
    void imgShow();        // 显示：clone 当前帧、画参考线、缩放后 imshow

    std::string getTime();

    // ---- 配置（由 camera_config.txt 载入）----
    Config cfg;

    // 以下为由 cfg 展开的常用字段，沿用原有名字
    int   exposureTime;       // 曝光时间 (us)
    float frameRate;
    bool  hardwareTrigger;    // true 时用硬件触发
    int   offsetY;
    int   offsetX;
    int   frameNum;
    int   imgNum;
    bool  showWindow;         // 是否开实时预览窗口
    std::string folderName;

    // 录像编码 fourcc，如 "XVID"(体积小，推荐) / "MJPG"(体积大) / "FFV1"(真无损但很慢)。
    // 来自配置文件的 codec 项，缺省 "XVID"。
    std::string codec;

    // 取帧线程的启停标志。主线程置 false 以结束取帧。
    // 用 atomic 是因为它被取帧线程读、主线程写。
    std::atomic<bool> isCapture;

    // ---- 相机与图像 ----
    Pylon::CBaslerUsbInstantCamera camera;
    Pylon::CDeviceInfo             info;
    Pylon::CGrabResultPtr          ptrGrabResult;
    Pylon::CPylonImage             pylonImg;
    Pylon::CImageFormatConverter   formatConverter;

    // 共享帧：取帧线程写，录像/显示/推理线程读。永远在 g_shared_mutex 保护下访问。
    cv::Mat opencvImg;

    cv::VideoWriter video;
};

#endif // !_TALK2CAMERA_H
