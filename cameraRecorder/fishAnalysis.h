#pragma once
#ifndef _FISH_ANALYSIS_H
#define _FISH_ANALYSIS_H

#include <atomic>

#include <opencv2/core/core.hpp>

#include "Talk2Camera.h"

// 推理消费方。当前只是一个占位实现：把共享帧取出来，统计处理帧率。
// 日后接入 TensorRT / DLC / 传统图像处理算法时，只需填充 processFrame()。
//
// 约定：processFrame 一定是在**不持有共享锁**的情况下被调用的。
// 取帧、clone、放锁、再推理——慢操作绝不能压着锁做，否则会卡住取帧线程。

// 处理单帧。返回 true 表示本帧处理成功。
bool processFrame(const cv::Mat& frame);

// 推理线程主体。按 targetFps 节拍运行，直到 running 变为 false。
void runInferenceLoop(PylonCameraData& cam, const std::atomic<bool>& running,
                      double targetFps);

#endif // !_FISH_ANALYSIS_H
