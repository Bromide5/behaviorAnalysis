# -*- coding: utf-8 -*-
"""
录音 · 监听 · 波形 一体化程序
============================

三个模块共用一个音频流，各自可在配置、命令行或**运行时的界面复选框**中
开关：

    ┌──────────┐
    │ 音频回调 │──> 队列 ──> 写盘线程 ──> rec_<时间戳>.wav
    │ (实时)   │──> 队列 ──> 主线程绘图 ──> 波形窗口
    │          │──> 输出缓冲 ────────────> 耳机（监听）
    └──────────┘

为什么必须共用一个流
--------------------
ASIO 是**独占**的：同一张声卡同一时刻只能被一个流占用。分别开「录音流」和
「监听流」会直接失败。所以这里用一个 `sd.Stream`：回调里同时完成
「入队存盘」「入队绘图」「写输出缓冲」三件事。

必需 / 可选
-----------
- **录音**、**监听**、**绘图** 全部关闭没有意义，程序会拒绝启动。
- **监听**需要设备有输出通道。若设备只有输入（例如 MME 的输入端点），
  程序会打印警告并自动关掉监听，其余功能照常。
- 只开绘图不开录音时，程序只显示波形，不产生文件。

用法
----
    python app.py                       # 录音（默认）
    python app.py --monitor             # 录音 + 监听
    python app.py --plot                # 录音 + 波形窗口（窗口内可切换录音/监听）
    python app.py --plot --monitor      # 三个全开
    python app.py --no-record --monitor # 只监听
    python app.py --no-record --plot    # 只看波形
    python app.py --list                # 列出设备
    python app.py --duration 60         # 60 秒后自动退出

运行时切换
----------
开 `--plot` 时，窗口底部有「录音」「监听」两个复选框，点一下即时生效：
录音关闭时会**正常收尾并写出当前文件**，再打开会另起一个新文件，
每个文件都有独立的元数据。这样长时间实验可以按 trial 分段。
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
import wave

import numpy as np

# 必须在 import sounddevice 之前置位，否则 ASIO 设备整个不会出现。
# （record.py 在模块导入时也会设，这里显式写一次以防单独运行本文件时漏掉。）
os.environ.setdefault("SD_ENABLE_ASIO", "1")

import sounddevice as sd  # noqa: E402

# 复用 record.py 里已经验证过的 24-bit 打包，避免两份实现走样
from mixer import MonitorMixer  # noqa: E402
from record import SAMPWIDTH, float_to_pcm24  # noqa: E402

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "app_config.txt")

PLOT_FPS = 25


# ===========================================================================
# 配置
# ===========================================================================

class Config:
    def __init__(self) -> None:
        # ---- 模块开关 ----
        self.enable_record = True
        self.enable_monitor = True
        self.enable_plot = True

        # ---- 设备 ----
        self.host_api = "ASIO"
        self.device_name = ""
        self.channels = 2
        self.samplerate = 48000
        self.blocksize = 0

        # ---- 录音 ----
        self.path = "./"
        self.base_name = "rec"
        self.queue_seconds = 10.0

        # ---- 监听 ----
        self.monitor_gain_db = 0.0
        self.monitor_out_channels = 2
        # auto = 单路有信号时复制到两耳（话筒只接 Input 1 的常见情况）
        self.monitor_mode = "auto"

        # ---- 显示 ----
        self.plot_window_s = 0.05     # 波形横轴时长(秒)
        self.show_level = True
        self.level_interval = 0.25

        # ---- 运行 ----
        self.duration = 0.0


_TRUTHY = ("true", "1", "yes", "on", "是")
_FALSY = ("false", "0", "no", "off", "否")


def _bool(v: str):
    s = v.strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return None


_KEYS = {
    "enablerecord": ("enable_record", bool),
    "enablemonitor": ("enable_monitor", bool),
    "enableplot": ("enable_plot", bool),
    "hostapi": ("host_api", str),
    "devicename": ("device_name", str),
    "channels": ("channels", int),
    "samplerate": ("samplerate", int),
    "blocksize": ("blocksize", int),
    "path": ("path", str),
    "basename": ("base_name", str),
    "queueseconds": ("queue_seconds", float),
    "monitorgaindb": ("monitor_gain_db", float),
    "monitoroutchannels": ("monitor_out_channels", int),
    "monitormode": ("monitor_mode", str),
    "plotwindows": ("plot_window_s", float),
    "showlevel": ("show_level", bool),
    "levelinterval": ("level_interval", float),
    "duration": ("duration", float),
}


def load_config(cfg: Config, filename: str) -> bool:
    try:
        fh = open(filename, "r", encoding="utf-8")
    except OSError:
        return False
    with fh:
        for lineno, raw in enumerate(fh, 1):
            t = raw.split("#", 1)[0].strip()
            if not t:
                continue
            if "=" not in t:
                print(f"[配置] 第 {lineno} 行缺少 '='，已跳过")
                continue
            key, val = t.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            spec = _KEYS.get(key)
            if spec is None:
                print(f"[配置] 第 {lineno} 行未知键，已忽略: {key}")
                continue
            attr, typ = spec
            try:
                if typ is bool:
                    b = _bool(val)
                    if b is None:
                        raise ValueError("需要 true/false")
                    setattr(cfg, attr, b)
                elif typ is str:
                    setattr(cfg, attr, val)
                else:
                    setattr(cfg, attr, typ(val))
            except ValueError as e:
                print(f"[配置] 第 {lineno} 行取值无效 ({key} = {val}): {e}，沿用原值")
    return True


def write_default_config(cfg: Config, filename: str) -> bool:
    text = f"""# 录音 · 监听 · 波形 一体化程序配置
# 格式：键 = 值   '#' 后为注释   改完直接重跑程序即可生效
# 命令行参数优先级高于本文件。

# ---------- 模块开关 ----------
# 三个都关掉没有意义，程序会拒绝启动
enableRecord = {"true" if cfg.enable_record else "false"}
enableMonitor = {"true" if cfg.enable_monitor else "false"}
enablePlot = {"true" if cfg.enable_plot else "false"}

# ---------- 设备 ----------
# 主机 API：ASIO / Windows WASAPI / MME ...
# 只有 ASIO 能同时收发（双工）；监听必须用 ASIO。
hostApi = {cfg.host_api}

# 设备名子串；留空 = 该主机 API 下第一个可用设备
deviceName = {cfg.device_name}

# 录制通道数
channels = {cfg.channels}

# 采样率。ASIO 下要与 Focusrite 控制里的设置一致
samplerate = {cfg.samplerate}

# 回调块大小；0 = 自动。录/监听想要低延迟可设 64 / 128 / 256
blocksize = {cfg.blocksize}

# ---------- 录音 ----------
# 保存根目录与文件名前缀：<path>/<baseName>/<baseName>_<时间戳>.wav
path = {cfg.path}
baseName = {cfg.base_name}

# 写盘队列容量（秒）。磁盘慢时调大
queueSeconds = {cfg.queue_seconds}

# ---------- 监听 ----------
# 监听增益(dB)。只影响耳机音量，不影响录音电平。
# 信号太小请拧声卡输入增益，别靠这里放大。
monitorGainDb = {cfg.monitor_gain_db}

# 监听输出通道数（Solo 的耳机/音箱是 2 通道）
monitorOutChannels = {cfg.monitor_out_channels}

# 监听映射方式：
#   auto   = 自动。某一路有信号、另一路只有底噪时，把有信号的那路复制到两耳
#            （话筒只接 Input 1 时的常见情况）；两路都有信号则按立体声走
#   stereo = 始终 1:1 映射（左->左，右->右）
#   mono   = 始终把 Input 1 复制到两耳
monitorMode = {cfg.monitor_mode}

# ---------- 显示 ----------
# 波形窗口横轴时长（秒）。0.05 表示每屏显示 50 ms
plotWindowS = {cfg.plot_window_s}

# 是否在控制台打印实时电平表
showLevel = {"true" if cfg.show_level else "false"}

# 电平表刷新间隔（秒）
levelInterval = {cfg.level_interval}

# ---------- 运行 ----------
# 运行秒数；<=0 表示一直运行到回车 / Ctrl+C / 关闭窗口
duration = {cfg.duration}
"""
    try:
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        return False
    return True


def print_config(cfg: Config) -> None:
    def yn(b):
        return "开" if b else "关"

    print("----- 生效配置 -----")
    print(f"  录音        : {yn(cfg.enable_record)}"
          + (f"  -> {os.path.join(cfg.path, cfg.base_name)}/" if cfg.enable_record else ""))
    print(f"  监听        : {yn(cfg.enable_monitor)}"
          + (f"  (增益 {cfg.monitor_gain_db:+g} dB, {cfg.monitor_mode})"
             if cfg.enable_monitor else ""))
    print(f"  波形        : {yn(cfg.enable_plot)}"
          + (f"  (窗宽 {cfg.plot_window_s*1000:.0f} ms)" if cfg.enable_plot else ""))
    print(f"  主机 API    : {cfg.host_api}")
    print(f"  设备        : {cfg.device_name or '(第一个可用)'}")
    print(f"  采样率      : {cfg.samplerate} Hz, {cfg.channels} ch")
    print(f"  块大小      : {cfg.blocksize or '自动'}")
    print(f"  时长        : " + (f"{cfg.duration:g} s" if cfg.duration > 0 else "手动停止"))
    print("--------------------")


# ===========================================================================
# 电平表
# ===========================================================================

def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


class LevelMeter:
    def __init__(self, channels: int) -> None:
        self.channels = channels
        self._lock = threading.Lock()
        self._peak = np.zeros(channels)
        self._sumsq = np.zeros(channels)
        self._count = 0
        self._clips = 0

    def update(self, block: np.ndarray) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            np.maximum(self._peak, np.max(np.abs(block), axis=0), out=self._peak)
            self._sumsq += np.sum(np.square(block, dtype=np.float64), axis=0)
            self._count += block.shape[0]
            self._clips += int(np.count_nonzero(np.abs(block) >= 0.9995))
        finally:
            self._lock.release()

    def take(self):
        with self._lock:
            rms = np.sqrt(self._sumsq / max(self._count, 1))
            out = (self._peak.copy(), rms, self._clips, self._count)
            self._peak[:] = 0.0
            self._sumsq[:] = 0.0
            self._count = 0
            self._clips = 0
            return out


# ===========================================================================
# 分段写盘线程
# ===========================================================================

class SegmentWriter(threading.Thread):
    """单线程串行处理「开文件 / 写数据 / 关文件」命令。

    所有操作走同一个 FIFO 队列，因此顺序天然正确：start 一定先于其后的
    data 被处理，stop 一定后于之前的数据。不能边读边关，否则会丢尾巴。
    """

    def __init__(self, samplerate: int, channels: int, maxsize: int) -> None:
        super().__init__(name="segment-writer", daemon=True)
        self.samplerate = samplerate
        self.channels = channels
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.error: Exception | None = None
        self.on_closed = None      # 回调：写入一个 segment 的元数据后调用

        self._wf = None
        self._path = ""
        self._meta: dict = {}
        self._frames = 0
        self._dropped = 0

    # ---- 供主线程调用 ----
    def cmd_start(self, path: str) -> None:
        self.q.put(("start", path, time.time()))

    def cmd_stop(self) -> None:
        self.q.put(("stop",))

    def cmd_quit(self) -> None:
        self.q.put(("quit",))

    def feed(self, block: np.ndarray) -> bool:
        """实时线程调用。队列满时丢块并返回 False，绝不阻塞。"""
        try:
            self.q.put_nowait(("data", block))
            return True
        except queue.Full:
            self._dropped += 1
            return False

    @property
    def dropped(self) -> int:
        return self._dropped

    # ---- 写盘线程 ----
    def run(self) -> None:
        try:
            while True:
                item = self.q.get()
                kind = item[0]
                if kind == "data":
                    if self._wf is not None:
                        self._wf.writeframes(float_to_pcm24(item[1]))
                        self._frames += item[1].shape[0]
                elif kind == "start":
                    self._open(item[1], item[2])
                elif kind == "stop":
                    self._close()
                elif kind == "quit":
                    self._close()
                    break
        except Exception as e:  # noqa: BLE001
            self.error = e
            try:
                self._close()
            except Exception:  # noqa: BLE001
                pass

    def _open(self, path: str, unix: float) -> None:
        self._close()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._wf = wave.open(path, "wb")
        self._wf.setnchannels(self.channels)
        self._wf.setsampwidth(SAMPWIDTH)
        self._wf.setframerate(self.samplerate)
        self._path = path
        self._frames = 0
        self._meta = {
            "file": os.path.basename(path),
            "start_unix": unix,
            "start_time_local": dt.datetime.fromtimestamp(unix).isoformat(
                timespec="milliseconds"),
            "samplerate": self.samplerate,
            "channels": self.channels,
            "bits_per_sample": SAMPWIDTH * 8,
        }

    def _close(self) -> None:
        if self._wf is None:
            return
        wf, self._wf = self._wf, None
        wf.close()
        meta = dict(self._meta)
        meta["frames"] = self._frames
        meta["duration_s"] = round(self._frames / self.samplerate, 6)
        try:
            with open(os.path.splitext(self._path)[0] + ".json", "w",
                      encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass
        if self.on_closed is not None:
            self.on_closed(meta)


# ===========================================================================
# 主程序
# ===========================================================================

class App:
    def __init__(self, cfg: Config, device: int) -> None:
        self.cfg = cfg
        self.device = device
        self.channels = cfg.channels
        self.samplerate = cfg.samplerate

        self.recording = cfg.enable_record
        self.monitoring = cfg.enable_monitor
        self.plotting = cfg.enable_plot
        self.monitor_gain = float(10.0 ** (cfg.monitor_gain_db / 20.0))
        self.mixer = None

        self.stop_evt = threading.Event()
        self.status_flags: set[str] = set()
        self.stream = None
        self.writer: SegmentWriter | None = None
        self.meter = LevelMeter(self.channels) if cfg.show_level else None
        self.stream_can_output = False   # 当前流是否建成了双工（可运行时开监听）
        self._suppress_check = False     # 防止 set_active 递归触发回调

        # 绘图队列：只保留最新几块，画不过来就丢旧的
        self.plot_q: queue.Queue = queue.Queue(maxsize=4)
        self.plot_buf: np.ndarray | None = None
        self._plot_line = None
        self._plot_fig = None
        self._plot_ax = None
        self._plot_title = None
        self._check = None
        self._plot_last = 0.0

        self.segments_written = 0
        self.t0 = time.monotonic()

    # ---------------------------------------------------------------- 设备
    def check_output(self) -> bool:
        d = sd.query_devices(self.device)
        api = sd.query_hostapis()[d["hostapi"]]["name"]
        if d["max_output_channels"] < 1:
            print(f"[警告] 设备 [{self.device}] {api} / {d['name']} 没有输出通道，"
                  f"监听已自动关闭。")
            print(f"       在 {api} 下 Focusrite 的输入和输出是两个独立端点；"
                  f"监听需要 ASIO。")
            return False
        return True

    # ---------------------------------------------------------------- 回调
    #
    # sounddevice 按流类型传不同数量的位置参数：
    #   输入流  -> callback(indata, frames, time, status)
    #   双工流  -> callback(indata, outdata, frames, time, status)
    # 必须用 *args 兼容两者，否则纯输入流下参数会错位（frames 被当成 outdata）。
    def _callback(self, indata, *args) -> None:
        # sounddevice 不做签名校验，直接把位置参数转进来：
        #   输入流  (indata, frames, time, status)          -> args 长度 3
        #   双工流  (indata, outdata, frames, time, status)  -> args 长度 4
        if len(args) == 4:
            outdata, _frames, _time, status = args
        else:
            outdata = None
            _frames, _time, status = args

        if status:
            self.status_flags.add(str(status))
        if self.meter is not None:
            self.meter.update(indata)

        # 录音：存的是**未加增益**的原始信号
        if self.recording and self.writer is not None:
            self.writer.feed(indata.copy())

        # 绘图：只取第一路
        if self.plotting:
            try:
                self.plot_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        # 监听：由 mixer 决定映射（单路有信号时自动复制到两耳）
        if outdata is not None:
            if self.monitoring and self.mixer is not None:
                self.mixer.process(indata, outdata)
            else:
                # 双工流建好后输出通道一直存在。监听关掉时必须显式清零：
                # PortAudio 不保证回调里的 outdata 是零，未写入的缓冲区会被
                # 反复送到耳机（听起来像持续的嗡声/循环噪声）。
                outdata.fill(0.0)

    # ---------------------------------------------------------------- 建流
    def open(self) -> None:
        dev = sd.query_devices(self.device)
        n_in = min(self.channels, int(dev["max_input_channels"]))
        if n_in < 1:
            raise SystemExit(f"[错误] 设备 [{self.device}] {dev['name']} 没有输入通道。")
        if n_in != self.channels:
            print(f"[提示] 设备只有 {n_in} 个输入通道，按 {n_in} 走")
            self.channels = n_in

        if self.monitoring and not self.check_output():
            self.monitoring = False

        n_out = 0
        if self.monitoring:
            n_out = min(self.cfg.monitor_out_channels,
                        int(dev["max_output_channels"]))
            self.monitoring = n_out > 0

        if self.recording:
            maxsize = max(4, int(self.cfg.queue_seconds * self.samplerate
                                 / max(self.cfg.blocksize or 512, 1)))
            self.writer = SegmentWriter(self.samplerate, self.channels, maxsize)
            self.writer.on_closed = self._on_segment_closed
            self.writer.start()

        print(f"[设备] [{self.device}] {dev['name']}  "
              f"{self.samplerate} Hz, {self.channels} 入"
              + (f" / {n_out} 出" if n_out else "")
              + f", 块 {self.cfg.blocksize or '自动'}")

        # 预检：采样率/通道数不被驱动接受时给出人话，而不是底层 PaErrorCode
        try:
            sd.check_input_settings(device=self.device, channels=self.channels,
                                    samplerate=self.samplerate, dtype="float32")
            if self.monitoring:
                sd.check_output_settings(device=self.device, channels=n_out,
                                         samplerate=self.samplerate,
                                         dtype="float32")
        except Exception as e:  # noqa: BLE001
            hint = (f"\n        设备默认采样率 {dev['default_samplerate']:.0f} Hz，"
                    f"可用输入 {dev['max_input_channels']} 通道。")
            if self.cfg.host_api.upper() == "ASIO":
                hint += ("\n        ASIO 下采样率必须与 Focusrite 控制里的设置一致"
                         "（本机当前是 44100）。")
            raise SystemExit(f"[错误] 设备不接受当前设置: {e}{hint}\n"
                             f"       试 --samplerate 44100")

        if self.monitoring:
            self.mixer = MonitorMixer(
                in_ch=self.channels, out_ch=n_out,
                gain=self.monitor_gain, mode=self.cfg.monitor_mode,
                samplerate=self.samplerate)
            self.stream = sd.Stream(
                device=self.device,
                channels=(self.channels, n_out),
                samplerate=self.samplerate,
                dtype="float32",
                blocksize=self.cfg.blocksize,
                latency="low",
                callback=self._callback,
            )
        else:
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                dtype="float32",
                blocksize=self.cfg.blocksize,
                callback=self._callback,
            )
        # 流一旦建成，输出通道数就固定了。运行时的监听开关只能在这个
        # 前提下工作，所以这里如实记录，供 set_monitoring 判断。
        self.stream_can_output = self.monitoring

    # ---------------------------------------------------------------- 录音控制
    def _new_path(self) -> str:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(self.cfg.path, self.cfg.base_name)
        p = os.path.join(d, f"{self.cfg.base_name}_{stamp}.wav")
        n = 1
        while os.path.exists(p):
            p = os.path.join(d, f"{self.cfg.base_name}_{stamp}_{n}.wav")
            n += 1
        return p

    def set_recording(self, on: bool) -> None:
        if self.writer is None:
            print("[录音] 未启用写盘线程，忽略")
            return
        if on == self.recording:
            return
        if on:
            # 先下发 start，再置标志：FIFO 保证 start 先被处理，
            # 之后入队的数据一定落在已打开的文件里。
            self.writer.cmd_start(self._new_path())
            self.recording = True
            print(f"\n[录音] 开始")
        else:
            self.recording = False
            self.writer.cmd_stop()
            print(f"\n[录音] 停止")

    def set_monitoring(self, on: bool) -> None:
        # 流的通道数在建立时就固定了，运行中无法增减输出通道。
        # 所以只在「本来就建了双工流」的前提下切换有效。
        if on and not self.stream_can_output:
            print("[监听] 当前流没有输出通道，无法开启。"
                  "请在启动时加 --monitor（或配置 enableMonitor = true）。")
            return
        if on == self.monitoring:
            return
        self.monitoring = on
        print(f"\n[监听] {'开' if on else '关'}")

    def _on_segment_closed(self, meta: dict) -> None:
        self.segments_written += 1
        print(f"\n[录音] 已保存 {meta['file']}  "
              f"({meta['duration_s']:.2f} s)")

    # ---------------------------------------------------------------- 绘图
    def _init_plot(self) -> None:
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons

        n = max(64, int(self.samplerate * self.cfg.plot_window_s))
        self.plot_buf = np.zeros(n)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        fig.subplots_adjust(bottom=0.22)
        self._plot_fig, self._plot_ax = fig, ax
        x = np.arange(n) / self.samplerate * 1000.0   # 毫秒
        (line,) = ax.plot(x, self.plot_buf, lw=0.8)
        self._plot_line = line
        ax.set_xlim(0, n / self.samplerate * 1000.0)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("幅度")
        ax.grid(True, alpha=0.3)
        self._plot_title = ax.set_title("")

        # 运行时开关。录音/监听可以随时切；绘图本身关掉就没窗口了，
        # 所以只给这两个复选框。
        cax = fig.add_axes([0.72, 0.03, 0.2, 0.11])
        self._check = CheckButtons(
            cax, ["录音", "监听"],
            [self.recording, self.monitoring])
        self._check.on_clicked(self._on_check)

        fig.canvas.mpl_connect("close_event", lambda e: self.stop_evt.set())
        plt.ion()

    def _on_check(self, label: str) -> None:
        if self._suppress_check:
            return
        from matplotlib.widgets import CheckButtons  # noqa: F401  （保持导入一致）
        idx = ["录音", "监听"].index(label)
        state = bool(self._check.get_status()[idx])

        if label == "录音":
            self.set_recording(state)
            if state != self.recording:
                # 写盘线程不可用时回弹，否则界面显示与真实状态不一致
                self._suppress_check = True
                self._check.set_active(idx)
                self._suppress_check = False
        else:
            self.set_monitoring(state)
            if state and not self.monitoring:
                # 当前流没有输出通道，开不了监听 -> 把复选框弹回去
                self._suppress_check = True
                self._check.set_active(idx)
                self._suppress_check = False

    def _update_plot(self) -> None:
        buf = self.plot_buf
        got = False
        while True:
            try:
                block = self.plot_q.get_nowait()
            except queue.Empty:
                break
            got = True
            k = min(len(block), len(buf))
            if k < len(buf):
                buf[:-k] = buf[k:]
                buf[-k:] = block[-k:]
            else:
                buf[:] = block[-len(buf):]
        if not got:
            return

        self._plot_line.set_ydata(buf)
        peak = float(np.max(np.abs(buf)))
        state = ("录音:开" if self.recording else "录音:关") + "  " + \
                ("监听:开" if self.monitoring else "监听:关")
        self._plot_title.set_text(
            f"{sd.query_devices(self.device)['name']}   "
            f"峰值 {db(peak):6.1f} dBFS   {state}")
        self._plot_fig.canvas.draw_idle()

    # ---------------------------------------------------------------- 主循环
    def run(self) -> int:
        print(f"  录音/监听/波形 = "
              f"{'开' if self.recording else '关'} / "
              f"{'开' if self.monitoring else '关'} / "
              f"{'开' if self.plotting else '关'}")

        if not self.plotting:
            self._run_headless()
        else:
            self._run_gui()

        self.close()

        # 结束时报告
        if self.segments_written:
            print(f"[记录] 共写出 {self.segments_written} 个文件")
        if self.writer is not None and self.writer.dropped:
            print(f"[警告] 写盘跟不上，丢弃 {self.writer.dropped} 块")
        if self.status_flags:
            print(f"[状态] {'; '.join(sorted(self.status_flags))}")
        return 0

    def _run_headless(self) -> None:
        self.open()
        if self.recording:
            self.writer.cmd_start(self._new_path())
        self.stream.start()
        self.t0 = time.monotonic()
        print("\n>>> 运行中。回车 / Ctrl+C 停止。\n")
        threading.Thread(target=self._wait_enter, daemon=True).start()

        next_tick = self.t0
        try:
            while not self.stop_evt.wait(0.05):
                now = time.monotonic()
                if self.cfg.duration > 0 and now - self.t0 >= self.cfg.duration:
                    break
                if self.meter is not None and now >= next_tick:
                    next_tick = now + self.cfg.level_interval
                    self._print_level(now)
        except KeyboardInterrupt:
            print("\n[主] 收到 Ctrl+C")

    def _run_gui(self) -> None:
        import matplotlib.pyplot as plt

        # 先 open：它会根据设备实际能力决定监听是否可行，
        # 之后再建界面，复选框的初始状态才是真实的。
        self.open()
        self._init_plot()
        if self.recording:
            self.writer.cmd_start(self._new_path())
        self.stream.start()
        self.t0 = time.monotonic()
        print("\n>>> 运行中。关窗口 / Ctrl+C 停止；可用复选框切换录音与监听。\n")

        interval = 1.0 / PLOT_FPS
        next_tick = self.t0
        try:
            while not self.stop_evt.is_set():
                if not plt.fignum_exists(self._plot_fig.number):
                    break
                now = time.monotonic()
                if self.cfg.duration > 0 and now - self.t0 >= self.cfg.duration:
                    break
                self._update_plot()
                if self.meter is not None and now >= next_tick:
                    next_tick = now + self.cfg.level_interval
                    self._print_level(now)
                self._plot_fig.canvas.flush_events()
                plt.pause(interval)
        except KeyboardInterrupt:
            print("\n[主] 收到 Ctrl+C")

    def _print_level(self, now: float) -> None:
        peak, rms, clips, _ = self.meter.take()
        parts = " ".join(f"{db(p):6.1f}" for p in peak)
        line = (f"\r[电平] 峰值 {parts} dBFS  "
                f"RMS {db(float(np.max(rms))):6.1f}  {now - self.t0:6.1f}s")
        if self.mixer is not None:
            line += f"  监听:{self.mixer.state}"
        if clips:
            line += f"  ⚠削波 {clips}"
        if float(np.max(peak)) < 1e-3 and now - self.t0 > 1.0:
            line += "  ← 几乎无信号: 拧输入增益/检查话筒"
        sys.stdout.write(line)
        sys.stdout.flush()

    def _wait_enter(self) -> None:
        try:
            line = sys.stdin.readline()
        except Exception:  # noqa: BLE001
            return
        if line == "":
            print("\n[提示] 标准输入已结束，回车停止不可用；请用 Ctrl+C 或 --duration。")
            return
        self.stop_evt.set()

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
        if self.writer is not None:
            self.writer.cmd_quit()
            self.writer.join(timeout=10)
            if self.writer.error is not None:
                print(f"[错误] 写盘线程异常: {self.writer.error}")


# ===========================================================================
# 命令行
# ===========================================================================

def resolve_device(cfg: Config) -> int:
    hostapis = sd.query_hostapis()
    api_idx = None
    for i, h in enumerate(hostapis):
        if h["name"].lower() == cfg.host_api.lower():
            api_idx = i
            break
    if api_idx is None:
        avail = ", ".join(h["name"] for h in hostapis)
        raise SystemExit(f"[错误] 没有 '{cfg.host_api}' 主机 API。可用: {avail}")

    devs = [d for d in sd.query_devices()
            if d["hostapi"] == api_idx and d["max_input_channels"] > 0]
    if not devs:
        raise SystemExit(
            f"[错误] 主机 API '{cfg.host_api}' 下没有输入设备。\n"
            f"       声卡未连接 / 驱动未注册 / 在本程序启动后才插上。\n"
            f"       先跑 python app.py --list 查看。")

    if cfg.device_name:
        hit = [d for d in devs if cfg.device_name.lower() in d["name"].lower()]
        if not hit:
            names = "\n".join(f"          [{d['index']}] {d['name']}" for d in devs)
            raise SystemExit(f"[错误] 找不到名字含 '{cfg.device_name}' 的设备。\n"
                             f"       可用:\n{names}")
        return int(hit[0]["index"])
    return int(devs[0]["index"])


def list_devices() -> None:
    hostapis = sd.query_hostapis()
    print("===== 音频设备 =====")
    for d in sd.query_devices():
        print(f"  {d['index']:>3}  {hostapis[d['hostapi']]['name']:<20} "
              f"in={d['max_input_channels']:<2} "
              f"out={d['max_output_channels']:<2} {d['name']}")
    print("\n监听需要 in>0 且 out>0 的设备（通常只有 ASIO）。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="录音 · 监听 · 波形 一体化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=CONFIG_FILE)
    p.add_argument("--save-config", default=None, metavar="PATH",
                   help="写出配置模板后退出")
    p.add_argument("--list", action="store_true", help="列出设备后退出")
    p.add_argument("--record", action=argparse.BooleanOptionalAction,
                   default=None, help="是否录音")
    p.add_argument("--monitor", action=argparse.BooleanOptionalAction,
                   default=None, help="是否监听（转发到耳机）")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction,
                   default=None, help="是否显示波形窗口")
    p.add_argument("--out", default=None, help="保存根目录")
    p.add_argument("--name", default=None, help="文件名前缀")
    p.add_argument("--device", default=None, help="设备名子串")
    p.add_argument("--host-api", default=None, help="主机 API 名")
    p.add_argument("--samplerate", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--blocksize", type=int, default=None)
    p.add_argument("--monitor-gain-db", type=float, default=None)
    p.add_argument("--monitor-mode", choices=["auto", "stereo", "mono"],
                   default=None,
                   help="监听映射：auto=单路有信号时复制到两耳")
    p.add_argument("--plot-window", type=float, default=None,
                   help="波形横轴时长(秒)")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--no-level", action="store_true", help="不打印电平表")
    return p


def main(argv=None) -> int:
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:  # noqa: BLE001
            pass
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args(argv)

    cfg = Config()
    if load_config(cfg, args.config):
        print(f"[配置] 已载入 {args.config}")
    else:
        print(f"[配置] 未找到 {args.config}，使用内置默认值")
        if write_default_config(cfg, args.config):
            print(f"[配置] 已生成模板: {args.config}")

    # 命令行覆盖
    for flag, attr in (("record", "enable_record"),
                       ("monitor", "enable_monitor"),
                       ("plot", "enable_plot")):
        v = getattr(args, flag)
        if v is not None:
            setattr(cfg, attr, v)
    for src in ("out", "name", "device", "host_api", "samplerate", "channels",
                "blocksize", "monitor_gain_db", "plot_window", "duration"):
        v = getattr(args, src)
        if v is not None:
            attr = {"out": "path", "name": "base_name", "device": "device_name",
                    "host_api": "host_api", "samplerate": "samplerate",
                    "channels": "channels", "blocksize": "blocksize",
                    "monitor_gain_db": "monitor_gain_db",
                    "plot_window": "plot_window_s",
                    "duration": "duration"}[src]
            setattr(cfg, attr, v)
    if args.monitor_mode is not None:
        cfg.monitor_mode = args.monitor_mode
    if args.no_level:
        cfg.show_level = False

    if args.save_config:
        ok = write_default_config(cfg, args.save_config)
        print(f"[配置] {'已写入 ' + args.save_config if ok else '写入失败'}")
        return 0 if ok else 1

    if args.list:
        list_devices()
        return 0

    if not (cfg.enable_record or cfg.enable_monitor or cfg.enable_plot):
        raise SystemExit("[错误] 录音、监听、波形三者全部关闭，没有可做的事。")

    print("===== 录音 · 监听 · 波形 =====")
    print_config(cfg)

    device = resolve_device(cfg)
    app = App(cfg, device)
    try:
        return app.run()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[错误] {e}")


if __name__ == "__main__":
    sys.exit(main())
