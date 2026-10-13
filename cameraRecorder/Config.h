#pragma once
#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

// 运行配置。
//
// 所有可调参数集中放在与 exe 同目录的文本文件里（缺省名 camera_config.txt），
// 程序本身不接收任何命令行参数。文件不存在时使用下面的内置默认值，并自动
// 生成一份带注释的模板供修改。
//
// 文件格式：每行 "键 = 值"，"#" 起首或行尾的注释会被忽略，空行忽略。
struct Config
{
    // ---- 相机 ----
    int         exposureTime   = 5000;               // 曝光时间 (us)
    bool        hardwareTrigger = false;             // false=自由运行；true=硬件触发
    std::string triggerSource  = "Line3";            // 硬件触发时的触发源
    int         offsetX        = 0;                  // 水平偏移 (px)
    int         offsetY        = 0;                  // 垂直偏移 (px)
    long long   deviceLinkThroughput = 400000000;    // 带宽上限 (Byte/s)
    std::string pfsFile        = "camera_params.pfs";// Pylon 参数文件；缺失则用上面各项

    // ---- 采集与录像 ----
    double      frameRate      = 30.0;               // 相机采集帧率 (fps)
    double      recordFps      = 0.0;                // 录像线程节拍；<=0 表示跟随 frameRate
    std::string path           = "./";               // 保存根目录
    std::string baseName       = "rec";              // 文件名前缀
    std::string codec          = "XVID";             // XVID / MJPG / FFV1（见 README）

    // ---- 显示 ----
    bool        showWindow     = true;               // 是否开实时预览窗口

    // 录像节拍的实际取值：recordFps>0 用它，否则跟随采集帧率
    double recordFpsEff() const { return recordFps > 0.0 ? recordFps : frameRate; }
};

// 从文件载入配置。返回 false 表示文件不存在或打不开（此时 cfg 保持内置默认值）。
bool loadConfig(Config& cfg, const std::string& filename);

// 把当前 cfg 写成一份带注释的模板（用于首次运行自动生成）。
bool writeDefaultConfig(const Config& cfg, const std::string& filename);

// 把生效的配置打印到控制台。
void printConfig(const Config& cfg);

#endif // !_CONFIG_H
