#include "Config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

namespace {

string trim(const string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

string toLower(string s)
{
    transform(s.begin(), s.end(), s.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return s;
}

bool parseBool(const string& v, bool& out)
{
    string s = toLower(trim(v));
    if (s == "true" || s == "1" || s == "yes" || s == "on" || s == "是")
    {
        out = true;
        return true;
    }
    if (s == "false" || s == "0" || s == "no" || s == "off" || s == "否")
    {
        out = false;
        return true;
    }
    return false;
}

} // namespace

bool loadConfig(Config& cfg, const string& filename)
{
    ifstream in(filename);
    if (!in.is_open())
        return false;

    string line;
    int lineno = 0;
    while (getline(in, line))
    {
        ++lineno;

        // 去掉行内注释（Pylon 的 pfs 文件名不会含 '#'，可安全截断）
        size_t hash = line.find('#');
        if (hash != string::npos)
            line = line.substr(0, hash);

        string t = trim(line);
        if (t.empty())
            continue;

        size_t eq = t.find('=');
        if (eq == string::npos)
        {
            cerr << "[配置] 第 " << lineno << " 行缺少 '='，已跳过: " << t << endl;
            continue;
        }

        string key = toLower(trim(t.substr(0, eq)));
        string val = trim(t.substr(eq + 1));

        try
        {
            if (key == "exposuretime" || key == "exposure_us")
                cfg.exposureTime = stoi(val);
            else if (key == "hardwaretrigger")
            {
                bool b;
                if (!parseBool(val, b))
                    throw runtime_error("需要 true/false");
                cfg.hardwareTrigger = b;
            }
            else if (key == "triggersource")
                cfg.triggerSource = val;
            else if (key == "offsetx")
                cfg.offsetX = stoi(val);
            else if (key == "offsety")
                cfg.offsetY = stoi(val);
            else if (key == "devicelinkthroughput")
                cfg.deviceLinkThroughput = stoll(val);
            else if (key == "pfsfile")
                cfg.pfsFile = val;
            else if (key == "framerate" || key == "fps")
                cfg.frameRate = stod(val);
            else if (key == "recordfps")
                cfg.recordFps = stod(val);
            else if (key == "path")
                cfg.path = val;
            else if (key == "basename")
                cfg.baseName = val;
            else if (key == "codec")
                cfg.codec = val;
            else if (key == "showwindow")
            {
                bool b;
                if (!parseBool(val, b))
                    throw runtime_error("需要 true/false");
                cfg.showWindow = b;
            }
            else
                cerr << "[配置] 第 " << lineno << " 行未知键，已忽略: " << key << endl;
        }
        catch (const exception& e)
        {
            cerr << "[配置] 第 " << lineno << " 行取值无效 (" << key << " = "
                 << val << "): " << e.what() << "，沿用原值" << endl;
        }
    }
    return true;
}

bool writeDefaultConfig(const Config& cfg, const string& filename)
{
    ofstream out(filename);
    if (!out.is_open())
        return false;

    out << "# 相机录像程序配置（与 exe 同目录）\n"
           "# 格式：键 = 值   '#' 后为注释   改完直接重跑程序即可生效\n"
           "# 本文件由程序在缺失时自动生成，可放心按需修改。\n"
           "\n"
           "# ---------- 相机 ----------\n"
           "# 曝光时间，微秒\n"
           "exposureTime = " << cfg.exposureTime << "\n"
           "\n"
           "# 触发模式：false = 自由运行（软件按帧率采）；true = 硬件触发\n"
           "hardwareTrigger = " << (cfg.hardwareTrigger ? "true" : "false") << "\n"
           "\n"
           "# 硬件触发时的触发源（相机 I/O 线名，如 Line3 / Line4）\n"
           "triggerSource = " << cfg.triggerSource << "\n"
           "\n"
           "# 画面偏移（ROI 起点），像素\n"
           "offsetX = " << cfg.offsetX << "\n"
           "offsetY = " << cfg.offsetY << "\n"
           "\n"
           "# USB 带宽上限，Byte/s\n"
           "deviceLinkThroughput = " << cfg.deviceLinkThroughput << "\n"
           "\n"
           "# Pylon Viewer 导出的参数文件。存在则先载入它，再用下面各项覆盖。\n"
           "# 留空或文件不存在则只用下面各项。\n"
           "pfsFile = " << cfg.pfsFile << "\n"
           "\n"
           "# ---------- 采集与录像 ----------\n"
           "# 相机采集帧率 (fps)\n"
           "frameRate = " << cfg.frameRate << "\n"
           "\n"
           "# 录像线程节拍 (fps)。<=0 表示跟随 frameRate。\n"
           "# 低于采集帧率可用于降采样录制、减小体积。\n"
           "recordFps = " << cfg.recordFps << "\n"
           "\n"
           "# 保存根目录与文件名前缀。录像写到 <path>/<baseName>/<baseName>_<时间戳>.avi\n"
           "path = " << cfg.path << "\n"
           "baseName = " << cfg.baseName << "\n"
           "\n"
           "# 录像编码：XVID(默认，体积最小) / MJPG(大) / FFV1(无损但慢)\n"
           "codec = " << cfg.codec << "\n"
           "\n"
           "# ---------- 显示 ----------\n"
           "# 是否开实时预览窗口。无显示环境（远程/无桌面）可设 false\n"
           "showWindow = " << (cfg.showWindow ? "true" : "false") << "\n";

    return static_cast<bool>(out);
}

void printConfig(const Config& cfg)
{
    cout << "----- 生效配置 -----\n"
         << "  曝光        : " << cfg.exposureTime << " us\n"
         << "  触发        : " << (cfg.hardwareTrigger ? "硬件触发 (" + cfg.triggerSource + ")"
                                                       : "自由运行")
         << "\n"
         << "  采集帧率    : " << cfg.frameRate << " fps\n"
         << "  录像帧率    : " << cfg.recordFpsEff() << " fps"
         << (cfg.recordFps > 0.0 ? "（独立设定）" : "（跟随采集）") << "\n"
         << "  输出        : " << cfg.path << cfg.baseName << "/\n"
         << "  编码        : " << cfg.codec << "\n"
         << "  预览窗口    : " << (cfg.showWindow ? "开" : "关") << "\n"
         << "--------------------\n";
}
