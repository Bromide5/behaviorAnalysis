# 相机录像程序（Basler Pylon + OpenCV）

用 Basler Pylon SDK 的 `CBaslerUsbInstantCamera` 打开相机，独立线程持续取帧，
写入共享的 `cv::Mat opencvImg`，由**录像 / 显示 / 推理**三个消费方在锁保护下各自取用。

参考程序：`D:\User\Bingrui_Wu\projects\HTFBA\HTFA_P_Cuda\HTFA_P_Cuda\`。
本程序沿用其架构，但剥掉了屏幕投影 / 继电器 / CUDA / TensorRT / MATLAB 依赖，
是可独立编译运行的最小版本。

## 架构

```
        [取帧线程 captureImg]  (Talk2Camera.cpp)
                 │ RetrieveResult → formatConverter(Mono8) → cv::Mat
                 │ g_shared_mutex.lock()          (独占写)
                 ▼
        共享 cv::Mat opencvImg   ← 唯一共享状态
                 │
     ┌───────────┼───────────┬───────────┐
     ▼           ▼           ▼
 lock_shared  lock_shared  lock_shared
 [录像]       [显示]       [推理]
 video.write  imshow       processFrame
 (main.cpp)   (Talk2Camera) (fishAnalysis.cpp)
```

- 写方（取帧线程）用 `g_shared_mutex.lock()` 独占访问。
- 读方用 `g_shared_mutex.lock_shared()`，**持锁期间只做一次 clone**，随即放锁，
  绝不压着锁做慢操作（如推理、写盘、imshow）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `Config.h` / `.cpp` | 配置文件解析与模板生成（`camera_config.txt`） |
| `lock.h` / `lock.cpp` | 全局 `std::shared_mutex g_shared_mutex` |
| `Talk2Camera.h` / `.cpp` | `PylonCameraData`：相机控制、取帧、录像、显示 |
| `fishAnalysis.h` / `.cpp` | 推理消费方，含空接口 `processFrame()` |
| `main.cpp` | 载入配置、起线程并管理退出 |

## 编译与运行

用 VS2022 打开 `cameraRecorder.sln`，选 x64 Debug/Release，F5；
或直接运行 `x64\Release\cameraRecorder.exe`。

**不接受任何命令行参数**——所有可调参数集中在与 exe 同目录的配置文件
`camera_config.txt` 里。该文件不存在时程序用内置默认值运行，并**自动生成一份
带注释的模板**，改完重跑即可生效（无需重新编译）。

**按回车结束录制**，输出 `<path>/<baseName>/<baseName>_<时间戳>.avi`（Mono8 灰度）。

运行前确保 pylon 与 OpenCV 的 DLL 在 PATH 中：
`C:\Program Files\Basler\pylon 5\Runtime\x64` 与 `D:\SoftWare\opencv\build\x64\vc15\bin`
（通常已在系统 PATH）。

## 配置文件 `camera_config.txt`

格式为 `键 = 值`，`#` 起首或行尾为注释，空行忽略。键名大小写不敏感；未知键与
取值无效的行会被跳过并在控制台提示，不影响其余配置。

| 键 | 含义 | 默认 |
| --- | --- | --- |
| `exposureTime` | 曝光时间 (us) | `5000` |
| `hardwareTrigger` | `true`=硬件触发，`false`=自由运行 | `false` |
| `triggerSource` | 硬件触发的触发源（线名，如 `Line3`) | `Line3` |
| `offsetX` / `offsetY` | 画面偏移（ROI 起点，px） | `0` |
| `deviceLinkThroughput` | USB 带宽上限 (Byte/s) | `400000000` |
| `pfsFile` | Pylon Viewer 导出的 `.pfs`；存在则先载入，再用上面各项覆盖 | `camera_params.pfs` |
| `frameRate` | 相机采集帧率 (fps) | `30` |
| `recordFps` | 录像线程节拍 (fps)；`<=0` 表示跟随 `frameRate` | `0` |
| `path` | 保存根目录 | `./` |
| `baseName` | 文件名前缀 | `rec` |
| `codec` | 录像编码：`XVID` / `MJPG` / `FFV1` | `XVID` |
| `showWindow` | 是否开实时预览窗口（无桌面环境设 `false`） | `true` |

`hardwareTrigger` 与 `showWindow` 接受 `true/false`、`yes/no`、`on/off`、`1/0`。

例：满速录制、只存不显（省 CPU）：

```ini
frameRate = 94.5
recordFps = 0        # 跟随采集
showWindow = false
codec = XVID
```

## 视频编码与体积

在真实画面（2048×1088 灰度，低对比度）上实测 200 帧的结果：

| 编码 | 压缩比 | 每帧 | 30 fps | 94.5 fps（满速） | 画质损失* |
| --- | --- | --- | --- | --- | --- |
| 未压缩 | 1.0x | 2.23 MB | 4.0 GB/min | 12.6 GB/min | 无 |
| **XVID**（默认） | **96.9x** | 0.023 MB | **41 MB/min** | **130 MB/min** | 1.94 |
| mp4v | 97.2x | 0.023 MB | 41 MB/min | 130 MB/min | 1.94 |
| MJPG | 17.4x | 0.128 MB | 231 MB/min | 727 MB/min | 1.71 |
| FFV1 | 2.2x | 1.02 MB | 1.8 GB/min | 5.8 GB/min | 0（真无损） |

\* 相对原始帧的平均绝对误差（灰度 0–255），越小越好。XVID 与 MJPG 的差异肉眼
不可辨，但体积相差近 6 倍。

推荐：**`XVID`（默认）**。若需要严格无损（例如事后要做精细的强度分析）用
`FFV1`，但它要 ~18 fps 的编码预算，满速 94.5 fps 录制时跟不上，只能降帧率用。
若追求更小体积且能接受轻微损失，可考虑外部 H.264 转码（见下）。

```ini
# 满速录制、体积最小
frameRate = 94.5
codec = XVID
```

```ini
# 无损，但编码慢，需降帧率
frameRate = 30
codec = FFV1
```

> **关于 H.264 / H.265**：OpenCV 自带的 FFmpeg 不含 H.264/HEVC 编码器
> （缺 `openh264-*.dll`，HEVC 走 MediaFoundation 报 COM 错误），实测无法直接写。
> 若确实需要，二选一：
> 1. 下载 `openh264-2.5.0-win64.dll` 放到 PATH，再把编码参数设为 `avc1`；
> 2. 先用 XVID 录，事后用独立 `ffmpeg` 转 H.264（可再小 3–5 倍）。
>
> 对分析用途，XVID 的体积已经足够小，通常不值得为 H.264 增加依赖。

## 相机参数

- 优先从 **`camera_params.pfs`**（`pfsFile` 指定）导入参数，该文件请用
  Pylon Viewer 调好参数后导出。存在时先载入它，再用配置文件中其余各项覆盖。
- 文件不存在时**静默回退**到配置文件里的各项（见上表），其内置默认值为：
  曝光 5000 us、帧率 30、自由运行、全画幅 2048x1088。
- 硬件触发在配置文件里设 `hardwareTrigger = true`、`triggerSource = Line3`（或
  你接线对应的线名）。

## 推理接口

`processFrame(const cv::Mat&)`（`fishAnalysis.cpp`）当前是占位实现，只统计帧率。
接入实际算法时把逻辑填进去即可。注意它总是在**不持有共享锁**的情况下被调用。

## 相对参考程序的修正

1. **共享帧必须 clone**：参考程序写
   `opencvImg = Mat(..., pylonImg.GetBuffer())` 是无拷贝引用，而 `pylonImg`
   会在下次 `Convert` 时被覆盖，导致消费方读到脏数据。本程序在锁内 `clone()`。
2. **`isCapture` 改为 `std::atomic<bool>`**：它被取帧线程读、主线程写，裸 `bool`
   是数据竞争。
3. **逐级创建目录**：`_mkdir` 只能建一级，本程序拆开逐段建，支持多级路径。
4. **去掉交互式询问**：不再开跑前 `cin` 问是否硬件触发，改为配置文件控制。
5. **`/utf-8` 编译选项**：源码含中文，中文 Windows 默认按 GBK 解析会报 C4819
   与"常量中有换行符"等一串错，故显式指定 UTF-8。
6. **相机先于 `PylonTerminate()` 析构**：原顺序是先 `PylonTerminate()` 再让相机
   对象析构，相机析构会访问已被回收的内部状态，退出时触发段错误（exit 139）。
   现把相机生命周期放进独立作用域，确保先析构、再终止 pylon。
7. **输出/中间目录显式设为绝对路径**：默认 `IntDir` 是相对路径，会随 MSBuild
   工作目录漂移，在项目目录下构建时会生成多余的嵌套 `cameraRecorder\` 子目录。

## 已知注意事项

- 本相机（acA2000-165umNIR）是**近红外**机型，**需要红外照明**才能成像。
  实测在照明关闭时整幅画面接近全黑（灰度 max≈3），这不是程序问题。
- 参数设置（曝光/帧率/触发）若个别项在机型上不可写，程序会打印警告并跳过，
  不会因此崩溃。
