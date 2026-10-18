# 录音程序（Focusrite / ASIO → 24-bit WAV）

从 Focusrite 声卡采集音频，边录边写 24-bit PCM WAV，可选实时监听。
与 `cameraRecorder`（录画面）独立运行，各自写自己的目录。

```
PortAudio 回调线程  --帧队列-->  写盘线程 --> rec_<时间戳>.wav
        |
        └--（可选）---> 耳机（监听，不落盘）
        |
        └--（可选）电平表，主线程定期打印
```

## 程序一览

| 程序 | 作用 |
| --- | --- |
| **`app.py`** | **一体化入口**：录音 / 监听 / 波形，三者的开关可在配置、命令行或运行时的复选框里改 |
| `record.py` | 只录音（可加 `--monitor` 边录边听） |
| `monitor.py` | 只监听，不录音 |
| `id.py` | 列出音频设备 |
| `soundRecord.py` | 实时波形图（只看，不保存） |
| `denoise.py` | 离线降噪：高通 / 工频陷波 / 谱减；原始文件不动 |

> **ASIO 是独占的**：同一张声卡同一时刻只能被一个流占用。所以这些程序
> **不能同时运行**。想让三件事同时发生，只能用 `app.py`——它把三者放进
> 同一个双工流里。

## 一体化程序 app.py

```bash
python app.py                        # 录音（默认）
python app.py --monitor              # 录音 + 监听
python app.py --plot                 # 录音 + 波形窗口
python app.py --plot --monitor       # 三个全开
python app.py --no-record --monitor  # 只监听
python app.py --no-record --plot     # 只看波形
python app.py --samplerate 44100     # 采样率必须与声卡当前设置一致
```

模块开关的优先级：**命令行 > 配置文件 > 内置默认值**。
配置文件是 `app_config.txt`（与程序同目录，缺失时自动生成）。

### 运行时切换

开 `--plot` 后，窗口底部有「录音」「监听」两个复选框，点一下即时生效：

- **关录音**：正常收尾，写出当前文件及其 `.json` 元数据。
- **重开录音**：另起一个新文件，每个文件独立计时。
  长时间实验可以借此按 trial 分段。
- **监听**：只有在启动时就建成了双工流（即启动时 `--monitor`）才能运行时
  开关——流的输出通道数在建立时固定，无法中途增减。启动时没开的话，
  复选框会被程序自动弹回，并打印原因。

### 为什么必须共用一个流

分别开「录音流」和「监听流」会直接失败（ASIO 独占）。`app.py` 用一个
`sd.Stream`，回调里同时完成三件事：

```
回调（实时线程）
  ├─ 入队 -> 写盘线程 -> rec_<时间戳>.wav
  ├─ 入队 -> 主线程绘图 -> 波形窗口
  └─ 写入输出缓冲 -> 耳机
```

绘图只能在主线程做，所以回调只负责把数据丢进队列；队列满了就丢旧块，
**绝不阻塞回调**（阻塞实时回调会爆音、丢帧）。

## 安装与运行

用 **`soundRecord`** conda 环境（里面装了 `sounddevice`）：

```bash
conda activate soundRecord

python record.py --list            # 先确认声卡认到没有
python record.py --duration 60     # 录 60 秒
python record.py                   # 回车停止
python record.py --monitor         # 边录边听
```

## 实时监听

```bash
# 一体化（推荐）
python app.py --plot --monitor

# 只监听，不录音。不加 --device 会自动挑可双工的设备
python monitor.py --gain-db 6
```

**为什么自动选中的是 ASIO**：MME / DirectSound / WASAPI 下，Focusrite 的输入和
输出被系统拆成**两个独立端点**，每个端点只具备单向能力，凑不出双工流——
所以 `--list` 里那些 `in=2 out=0` / `in=0 out=2` 的条目都不能用来监听。
只有 ASIO 那一条（`in=2 out=2`）可以。`monitor.py` 不指定 `--device` 时会
自动跳过这些单向端点，优先选 ASIO。

要点：

- **必须用耳机**。音箱 + 话筒会形成声学正反馈，几秒内啸叫。
- 声卡上若按下了 **DIRECT MONITOR**，请关掉——否则会听到两份（硬件一份、
  软件一份），有轻微梳状滤波。
- `--monitor-gain-db` 只影响**耳机**音量，**不影响录音电平**。录到的信号
  太小请拧声卡上的输入增益旋钮，软件放大只会把噪声一起放大。
- 延迟主要是块周期：`blocksize / samplerate × 2`。例如 44100 Hz、块 64
  约 3 ms。人耳对超过 20 ms 的延迟会觉察到回声感，想更跟手就把
  `blocksize` 设小（64 或 128）。
- 注意 `sounddevice` 对 ASIO 报告的 `stream.latency` 是固定占位值
  （本机恒为 47.9/49.9 ms，与块大小无关），**不可信**，程序已改为按块周期估算。

### 耳机只有左边有声音？

这不是驱动问题，而是**话筒只插在 Input 1**。2 通道输入按 1:1 映射到 2 通道
输出时，右耳拿到的是 Input 2 的信号——那里只有底噪（本机实测约 -92 dBFS），
听起来就是右边没声。既有录音的逐声道测量也印证了这一点：左声道 RMS
-16 ~ -44 dBFS，右声道 RMS 约 -102 dBFS，两声道相关系数接近 0。

三种处理方式：

| 做法 | 效果 |
| --- | --- |
| `monitorMode = auto`（默认） | 检测到只有一路有信号时，把它复制到两耳；两路都有信号（如立体声话筒）时按 1:1 走 |
| `monitorMode = mono` | 始终把 Input 1 复制到所有输出通道 |
| 硬件：把话筒换到 Input 2，或使用 2 路话筒 | 真正变成双声道，可保持 `stereo` |

`auto` 带迟滞（-55 dBFS 判为有信号、-70 dBFS 判为静音）和 0.3 s 驻留，
说话停顿不会让声音在左右耳之间来回跳。也可以用 `--monitor-mode stereo`
强制严格 1:1，或 `--monitor-mode mono` 强制单声道。

## 降噪（denoise.py）

实时链路只转发原始输入，软件降噪放在**离线后处理**：STFT 谱减需要逐帧
运算，放进音频回调会加延迟，也容易产生「音乐噪声」。

```bash
python denoise.py rec/rec_xxx.wav            # 输出 rec/rec_xxx_clean.wav
python denoise.py rec                        # 处理目录下全部 wav
python denoise.py rec/rec_xxx.wav --over 2.5 --floor-db -20
python denoise.py rec/rec_xxx.wav --no-spectral   # 只做高通 + 低通
python denoise.py rec/rec_xxx.wav --highpass 0 --lowpass 0  # 只做谱减
python denoise.py rec/rec_xxx.wav --noise-file noise.wav  # 指定纯噪声样本
```

处理顺序：

1. **高通 1 kHz**（默认）——切掉低频隆隆、交流声和目标信号以下的噪声。
2. **低通 20 kHz**（默认）——切掉高频嘶声；48 kHz 采样下只影响 20–24 kHz。
3. **谱减**（默认 `over=2.0`, `floor=-15 dB`）——自动取最安静的 16 个
   1 秒窗口估计噪声谱，逐帧做 `|X| - over·|N|`，再压谱地板。

本机 5 分 40 秒录音、默认参数实测：安静段 RMS 从 **-45.5 降到 -61.9 dBFS**
（降 16.4 dB），最响段从 -25.1 降到 -33.8 dBFS（降 8.7 dB——目标信号若
本身有低频成分，会被 1 kHz 高通一并削掉）。处理耗时约 10 秒。输出固定为
24-bit PCM，采样率/声道数与原文件一致，另写一份 `_clean.json` 记录参数和
电平。原始文件不会被修改或覆盖。

> 如果采集链路已经做了 1 kHz 高通 / 20 kHz 低通，就不要再叠一次：
> `--highpass 0 --lowpass 0`，只保留谱减。

高通会在瞬态上产生过冲：本机这段录音（输入峰值已经顶到 0 dBFS）经
1 kHz 高通后浮点峰值到 +5.5 dBFS，工具会按 24-bit 上限截幅并警告。
根治办法是录音时把声卡输入增益降下来，让峰值留 6 dB 余量。

参数怎么调：

- 底噪还在 → `--over` 往 2.5 调，或 `--floor-db` 往 -20 调。
- 听到流水声/鸟叫似的伪影 → 反方向调。
- 安静段和整体电平差不到 10 dB 时程序会警告：谱减可能连目标声一起削弱，
  此时先用 `--no-spectral` 只做滤波，或录一段纯噪声用 `--noise-file` 指定。

## 为什么需要 `SD_ENABLE_ASIO`

sounddevice 的 ASIO 支持**默认是关的**（ASIO SDK 授权不允许默认启用）。
不设这个变量，它会加载不带 ASIO 的 PortAudio DLL，ASIO 主机 API 会整个消失，
声卡驱动装得再对也扫不出来——这是最常见的「装了驱动却找不到 ASIO」的原因。

环境变量已在 `soundRecord` 环境里永久设好：

```bash
conda env config vars set SD_ENABLE_ASIO=1 -n soundRecord
```

`record.py` 顶部也有一行 `os.environ.setdefault("SD_ENABLE_ASIO", "1")` 兜底，
所以在别的环境里直接跑也不会踩这个坑。

## 配置

所有参数写在与本文件同目录的 `audio_config.txt`（键 = 值，`#` 后为注释），
命令行参数优先级更高。文件不存在时用内置默认值并自动生成一份模板。

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `hostApi` | `ASIO` | `ASIO` / `Windows WASAPI` / `Windows DirectSound` / `MME` |
| `deviceName` | 空 | 设备名子串；空 = 该主机 API 下第一个可录设备 |
| `channels` | `2` | 录制通道数 |
| `inputChannels` | 空 | ASIO 物理通道选择，如 `0,1`；填了以它为准 |
| `samplerate` | `48000` | 采样率。ASIO 下必须与 Focusrite Control 2 里设的一致 |
| `blocksize` | `0` | 回调帧数；`0` = 自动。爆音/丢帧可试 128 / 256 / 512 |
| `path` | `./` | 保存根目录 |
| `baseName` | `rec` | 文件名前缀 |
| `duration` | `0` | 录制秒数；`<=0` 表示回车 / Ctrl+C 手动停止 |
| `showLevel` | `true` | 是否打印实时电平表 |
| `queueSeconds` | `10` | 写盘队列容量（秒），磁盘慢时调大 |
| `monitor` | `false` | 是否边录边听（转发到耳机） |
| `monitorGainDb` | `0` | 监听增益(dB)，只影响耳机音量 |
| `monitorOutChannels` | `2` | 监听输出通道数 |
| `monitorMode` | `auto` | `auto` / `stereo` / `mono`，见上文「耳机只有左边有声音？」 |

输出写到 `<path>/<baseName>/<baseName>_<时间戳>.wav`。

## 输出

| 文件 | 内容 |
| --- | --- |
| `rec_<时间戳>.wav` | 24-bit PCM，采样率与通道数按配置 |
| `rec_<时间戳>.json` | 旁车元数据：**绝对起始时间** `start_unix`、设备、时长、丢块数、削波数等 |

`start_unix` 是用来**和视频对齐**的：`cameraRecorder` 的录像文件名里带时间戳，
两者的绝对起始时间一比，就能确定音视频的时间偏移。

## 与视频对齐

```python
import json, datetime as dt
meta = json.load(open("soundRecorder/rec/rec_20261017_175504.json"))
print(meta["start_unix"], meta["start_time_local"], meta["duration_s"])
```

## 关于延迟与丢帧

音频回调（实时线程）只做一次 `copy()` 和入队，24-bit 打包与磁盘 I/O 全在写盘
线程里做——在回调里做重活会直接导致爆音和丢帧。队列写满时**丢弃该块并计数**，
不阻塞回调；如果结束后报告 `丢弃块` 非零，说明磁盘跟不上，把 `queueSeconds`
调大，或换更快的盘。

## 常见问题

**ASIO 下一个设备都没有**
1. 声卡没连上（USB 松了 / 没供电 / 被别的程序独占）
2. 驱动没装，或装的是 32 位版（64 位 Python 需要 64 位 ASIO）
3. 声卡在程序启动**之后**才插上——PortAudio 只在启动时枚举一次，要重开程序

先 `python record.py --list` 看设备。急着录可以临时把 `hostApi` 改成
`Windows WASAPI` 顶一下（延迟高些，但不需要 ASIO）。

**打不开流 / 报采样率不被接受**
ASIO 下采样率必须与 Focusrite Control 2 里设置的一致，改完重开程序。

**回车停不下来**
stdin 是管道或文件时（比如被脚本调用），回车停止不可用，程序会提示；
这时请用 `--duration` 或 Ctrl+C。

## 已知限制

- 单文件录制，不做分段/自动切分。长时录制请留意磁盘空间：
  48 kHz / 2 ch / 24-bit ≈ **1 GB / 小时**。
- 不检测「静音」或「削波」自动告警，只在结束时汇总报告。
- `soundRecord.py` 是早期实时波形预览草稿（依赖未安装的 `matplotlib`，
  且 `import soundRecorder` 有误），与本程序无关，可忽略或删除。
