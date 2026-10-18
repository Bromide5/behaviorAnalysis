# -*- coding: utf-8 -*-
"""
斑马鱼行为实验录音程序
====================

从 Focusrite 声卡（ASIO）采集音频，边录边写 24-bit PCM WAV。

与 cameraRecorder 的分工
------------------------
cameraRecorder 录画面，本程序录声音，两者独立运行、各自写自己的目录。
每次录音额外写一个同名 `.json` 旁车文件，里面记录了**绝对起始时间**
（`start_unix`），后续可用它把音频与视频对齐。

架构
----
PortAudio 回调线程  --帧队列-->  写盘线程 --> rec_<时间戳>.wav
        |
        └--（可选）电平表，主线程定期打印

音频回调只做一次 `copy()` 和入队，24-bit 打包与磁盘 I/O 都在写盘线程里做，
避免在实时回调里阻塞导致爆音/丢帧。队列写满时丢弃该块并计数，不阻塞回调。

用法
----
    python record.py                 # 回车停止
    python record.py --duration 60   # 录 60 秒自动停
    python record.py --list          # 只列出设备，不录音

所有参数也可写在与本文件同目录的 `audio_config.txt` 里（键 = 值），
命令行参数优先级更高。该文件不存在时会用内置默认值并自动生成一份。
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

# PortAudio 的 ASIO 支持在 sounddevice 里是默认关闭的（ASIO SDK 授权所限），
# 必须在此处、即 import sounddevice 之前置位，才会加载 libportaudio*-asio.dll。
# 否则 ASIO 主机 API 会整个消失，声卡驱动装得再对也扫不出来。
os.environ.setdefault("SD_ENABLE_ASIO", "1")

import sounddevice as sd  # noqa: E402  （必须在设置环境变量之后）

from mixer import MonitorMixer  # noqa: E402

# 配置文件缺省名（与本文件同目录，不是当前工作目录）
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "audio_config.txt")

SAMPWIDTH = 3  # WAV 采样位宽（字节），3 = 24-bit
FULL_SCALE = 1 << 23  # 2**23 = 8388608


# ===========================================================================
# 配置
# ===========================================================================

class Config:
    """录音参数。字段名与配置文件里的键一一对应（比较时不区分大小写）。"""

    def __init__(self) -> None:
        # ---- 设备 ----
        self.host_api = "ASIO"        # 主机 API 名：ASIO / Windows WASAPI / MME ...
        self.device_name = ""         # 设备名子串；空 = 该主机 API 下第一个可录设备
        self.channels = 2             # 录多少通道
        self.input_channels = ""      # ASIO 物理通道选择，如 "0,1"；空 = 顺序取前 N 个
        self.samplerate = 48000       # 采样率
        self.blocksize = 0            # 每次回调帧数；0 = 交给 PortAudio/驱动决定

        # ---- 输出 ----
        self.path = "./"              # 保存根目录
        self.base_name = "rec"        # 文件名前缀
        self.duration = 0.0           # 录制时长(秒)；<=0 表示回车/Ctrl+C 手动停止

        # ---- 显示 ----
        self.show_level = True        # 是否打印实时电平表
        self.level_interval = 0.25    # 电平表刷新间隔(秒)

        # ---- 监听 ----
        # 录音的同时把输入转发到耳机。ASIO 独占，无法另开一个输出流，
        # 所以必须复用同一个双工流（见 monitor.py 的说明）。
        self.monitor = False
        self.monitor_gain_db = 0.0    # 只影响听到的音量，不影响录音
        self.monitor_out_channels = 2
        # auto = 单路有信号时复制到两耳（话筒只接 Input 1 的常见情况）
        self.monitor_mode = "auto"

        # ---- 缓冲 ----
        self.queue_seconds = 10.0     # 写盘队列容量(秒)；磁盘慢时给足缓冲

    # 设备串在 console 上的显示名
    def device_desc(self) -> str:
        return self.device_name or "(第一个可用)"


# ---- 配置文件读写（格式与 camera_config.txt 一致：键 = 值，# 后为注释）----

def _trim(s: str) -> str:
    return s.strip(" \t\r\n")


def _parse_bool(v: str):
    s = _trim(v).lower()
    if s in ("true", "1", "yes", "on", "是"):
        return True
    if s in ("false", "0", "no", "off", "否"):
        return False
    return None


# 配置文件键名 -> (Config 属性, 类型)
_KEYS = {
    "hostapi": ("host_api", str),
    "devicename": ("device_name", str),
    "channels": ("channels", int),
    "inputchannels": ("input_channels", str),
    "samplerate": ("samplerate", int),
    "blocksize": ("blocksize", int),
    "path": ("path", str),
    "basename": ("base_name", str),
    "duration": ("duration", float),
    "showlevel": ("show_level", bool),
    "levelinterval": ("level_interval", float),
    "queueseconds": ("queue_seconds", float),
    "monitor": ("monitor", bool),
    "monitorgaindb": ("monitor_gain_db", float),
    "monitoroutchannels": ("monitor_out_channels", int),
    "monitormode": ("monitor_mode", str),
}


def load_config(cfg: Config, filename: str) -> bool:
    """从文件载入配置。返回 False 表示文件不存在或打不开（cfg 保持默认值）。"""
    try:
        fh = open(filename, "r", encoding="utf-8")
    except OSError:
        return False

    with fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0]          # 去掉注释
            t = _trim(line)
            if not t:
                continue
            if "=" not in t:
                print(f"[配置] 第 {lineno} 行缺少 '='，已跳过: {t}")
                continue
            key, val = t.split("=", 1)
            key = _trim(key).lower()
            val = _trim(val)

            spec = _KEYS.get(key)
            if spec is None:
                print(f"[配置] 第 {lineno} 行未知键，已忽略: {key}")
                continue
            attr, typ = spec
            try:
                if typ is bool:
                    b = _parse_bool(val)
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
    """写一份带注释的模板。"""
    text = f"""# 录音程序配置（与本文件同目录）
# 格式：键 = 值   '#' 后为注释   改完直接重跑程序即可生效
# 本文件由程序在缺失时自动生成，可放心按需修改。

# ---------- 设备 ----------
# 主机 API：ASIO / Windows WASAPI / Windows DirectSound / MME
# 低延迟录音用 ASIO（需装 Focusrite 驱动）；声卡不在时可用 WASAPI 顶一下
hostApi = {cfg.host_api}

# 设备名子串（不必写全，能唯一匹配即可）。
# 留空 = 该主机 API 下第一个有输入通道的设备。
deviceName = {cfg.device_name}

# 录制通道数。Focusrite 双通道下 ASIO 设备一般暴露 2 通道起
channels = {cfg.channels}

# ASIO 物理通道选择，逗号分隔，如 0,1 表示取前两个输入。
# 留空 = 顺序取前 channels 个。填了就以这里为准（会覆盖 channels）。
inputChannels = {cfg.input_channels}

# 采样率。ASIO 下必须与 Focusrite Control 2 里设的一致，否则打不开流
samplerate = {cfg.samplerate}

# 每次回调帧数。0 = 交给 PortAudio/驱动决定（推荐）。
# 若出现爆音或丢帧，可试 128 / 256 / 512
blocksize = {cfg.blocksize}

# ---------- 输出 ----------
# 保存根目录与文件名前缀。录音写到 <path>/<baseName>/<baseName>_<时间戳>.wav
# 同目录还会生成同名 .json，记录绝对起始时间等，供与视频对齐
path = {cfg.path}
baseName = {cfg.base_name}

# 录制时长（秒）。<=0 表示由回车 / Ctrl+C 手动停止
duration = {cfg.duration}

# ---------- 显示 ----------
# 是否打印实时电平表（无桌面环境也能用，纯控制台输出）
showLevel = {"true" if cfg.show_level else "false"}

# 电平表刷新间隔（秒）
levelInterval = {cfg.level_interval}

# ---------- 缓冲 ----------
# 写盘队列容量（秒）。磁盘慢或网络盘时调大，内存占用 = 采样率*通道*4字节*该值
queueSeconds = {cfg.queue_seconds}

# ---------- 监听 ----------
# 录音同时把输入转发到耳机（边录边听）。
# 注意：ASIO 是独占的，monitor.py 不能与本程序同时运行；
#       要边录边听只能打开这里的开关。
# 用耳机听；用音箱 + 话筒会啸叫。声卡上若按了 DIRECT MONITOR，请关掉。
monitor = {"true" if cfg.monitor else "false"}

# 监听增益(dB)。只影响你听到的音量，不影响录音电平。
# 录到的信号太小请拧声卡输入增益旋钮，别靠这里放大。
monitorGainDb = {cfg.monitor_gain_db}

# 监听输出通道数（Solo 的耳机/音箱是 2 通道）
monitorOutChannels = {cfg.monitor_out_channels}

# 监听映射方式：
#   auto   = 自动。某一路有信号、另一路只有底噪时，把有信号的那路复制到两耳
#   stereo = 始终 1:1 映射
#   mono   = 始终把 Input 1 复制到两耳
monitorMode = {cfg.monitor_mode}
"""
    try:
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        return False
    return True


def print_config(cfg: Config) -> None:
    print("----- 生效配置 -----")
    print(f"  主机 API    : {cfg.host_api}")
    print(f"  设备        : {cfg.device_desc()}")
    print(f"  通道        : {cfg.channels}"
          + (f"（物理通道 {cfg.input_channels}）" if cfg.input_channels else ""))
    print(f"  采样率      : {cfg.samplerate} Hz")
    print(f"  块大小      : {cfg.blocksize or '自动'}")
    print(f"  输出        : {os.path.join(cfg.path, cfg.base_name)}/")
    print(f"  时长        : "
          + (f"{cfg.duration:g} s" if cfg.duration > 0 else "手动停止"))
    print(f"  电平表      : {'开' if cfg.show_level else '关'}")
    print(f"  监听        : "
          + (f"开（增益 {cfg.monitor_gain_db:+g} dB, {cfg.monitor_mode}）"
             if cfg.monitor else "关"))
    print("--------------------")


# ===========================================================================
# 设备解析
# ===========================================================================

def find_host_api(name: str):
    """按名字找主机 API，返回索引；找不到返回 None。"""
    for i, h in enumerate(sd.query_hostapis()):
        if h["name"].lower() == name.lower():
            return i
    return None


def resolve_device(cfg: Config) -> tuple[int, int]:
    """解析出 (设备索引, 主机API索引)。失败时抛 SystemExit 并给出可读的原因。"""
    hostapis = sd.query_hostapis()
    api_idx = find_host_api(cfg.host_api)
    if api_idx is None:
        avail = ", ".join(h["name"] for h in hostapis)
        raise SystemExit(f"[错误] 没有 '{cfg.host_api}' 这个主机 API。\n"
                         f"       当前可用: {avail}")

    devs = [d for d in sd.query_devices()
            if d["hostapi"] == api_idx and d["max_input_channels"] > 0]

    if not devs:
        msg = [f"[错误] 主机 API '{cfg.host_api}' 下没有可用的输入设备。"]
        if cfg.host_api.upper() == "ASIO":
            msg += [
                "",
                "        ASIO 下一个设备都没有，通常是以下之一：",
                "          1. 声卡没连上电脑（USB 松了 / 没供电 / 被别的程序独占）",
                "          2. 驱动没装或装的是 32 位版（64 位 Python 需要 64 位 ASIO）",
                "          3. 声卡在本程序启动后才插上——PortAudio 只在启动时枚举一次",
                "",
                "        先跑  python record.py --list  看设备；",
                "        若是急着录，可把配置里的 hostApi 改成 Windows WASAPI 顶一下。",
            ]
        raise SystemExit("\n".join(msg))

    if cfg.device_name:
        hit = [d for d in devs if cfg.device_name.lower() in d["name"].lower()]
        if not hit:
            names = "\n".join(f"          [{d['index']}] {d['name']}" for d in devs)
            raise SystemExit(
                f"[错误] 在 '{cfg.host_api}' 下找不到名字含 "
                f"'{cfg.device_name}' 的输入设备。\n"
                f"       该主机 API 下可用的有:\n{names}")
        dev = hit[0]
    else:
        dev = devs[0]

    return dev["index"], api_idx


def resolve_channels(cfg: Config, device: int):
    """定下最终 (录制通道数, ASIO 通道选择设置)。

    ASIO 下若填了 inputChannels，就按选择器个数走——必须在开流**之前**定死，
    否则队列容量、电平表、WAV 头会各自按不同通道数建，写出来的文件是坏的。
    """
    extra = None
    channels = cfg.channels
    n_avail = int(sd.query_devices(device)["max_input_channels"])

    if cfg.host_api.upper() == "ASIO" and cfg.input_channels:
        try:
            sel = [int(s) for s in cfg.input_channels.replace("，", ",").split(",")
                   if s.strip() != ""]
        except ValueError:
            raise SystemExit(f"[错误] inputChannels 只能填逗号分隔的整数，"
                             f"当前为 '{cfg.input_channels}'")
        if sel:
            for c in sel:
                if c < 0 or c >= n_avail:
                    raise SystemExit(
                        f"[错误] inputChannels 里的 {c} 超出范围"
                        f"（该设备可用物理通道 0..{n_avail - 1}）")
            if len(sel) != channels:
                print(f"[提示] inputChannels 有 {len(sel)} 个通道"
                      f"（{sel}），录制通道数按 {len(sel)} 走，"
                      f"配置里的 channels={channels} 被忽略")
            channels = len(sel)
            extra = sd.AsioSettings(channel_selectors=sel)

    if channels > n_avail:
        raise SystemExit(f"[错误] 要录 {channels} 通道，"
                         f"但设备只提供 {n_avail} 个输入通道")

    return channels, extra


def print_devices() -> None:
    """列出所有设备，并单独说明 ASIO 的情况。"""
    hostapis = sd.query_hostapis()
    print("===== 音频设备 =====")
    for i, h in enumerate(hostapis):
        print(f"[{i}] {h['name']}  ({len(h['devices'])} 个设备)")
    print()
    for d in sd.query_devices():
        api = hostapis[d["hostapi"]]["name"]
        kind = "in=%d out=%d" % (d["max_input_channels"], d["max_output_channels"])
        print(f"  {d['index']:>3}  {api:<22} {kind:<16} {d['name']}")

    print()
    if "ASIO" in [h["name"] for h in hostapis]:
        asio = [d for d in sd.query_devices()
                if hostapis[d["hostapi"]]["name"] == "ASIO"
                and d["max_input_channels"] > 0]
        if asio:
            print(">>> ASIO 可用:")
            for d in asio:
                print(f"      {d['name']}  "
                      f"(in={d['max_input_channels']}, "
                      f"默认采样率={d['default_samplerate']:.0f} Hz)")
        else:
            print(">>> ASIO 主机 API 存在，但一个输入设备都没有 "
                  "—— 声卡未连接 / 驱动未注册。")
    else:
        print(">>> 没有 ASIO 主机 API。若已装 Focusrite 驱动，"
              "多半是 SD_ENABLE_ASIO 没生效（见 README）。")


# ===========================================================================
# 24-bit 打包与写盘
# ===========================================================================

def float_to_pcm24(x: np.ndarray) -> bytes:
    """(n, ch) float32，取值 [-1, 1] -> 24-bit 小端 PCM 字节串。

    先截幅再缩放：满量程映射到 [-2^23, 2^23-1]。取 4 字节小端的低 3 字节，
    正好是 24 位补码的补码表示，无需逐样本处理。
    """
    y = np.clip(x, -1.0, 1.0)
    i = np.rint(y * float(FULL_SCALE)).astype(np.int32)
    np.clip(i, -FULL_SCALE, FULL_SCALE - 1, out=i)
    return i.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


class WavWriter(threading.Thread):
    """写盘线程：从队列取 float32 块，转 24-bit 后追加写入 WAV。"""

    def __init__(self, path: str, samplerate: int, channels: int,
                 q: "queue.Queue", stop_evt: threading.Event) -> None:
        super().__init__(name="wav-writer", daemon=True)
        self.path = path
        self.samplerate = samplerate
        self.channels = channels
        self.q = q
        self.stop_evt = stop_evt
        self.frames_written = 0
        self.blocks_written = 0
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            with wave.open(self.path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(SAMPWIDTH)
                wf.setframerate(self.samplerate)
                while True:
                    try:
                        block = self.q.get(timeout=0.2)
                    except queue.Empty:
                        # 队列空了：流停下且收到哨兵就退出，否则继续等
                        if self.stop_evt.is_set():
                            break
                        continue
                    if block is None:      # 哨兵：正常收尾
                        break
                    wf.writeframes(float_to_pcm24(block))
                    self.frames_written += block.shape[0]
                    self.blocks_written += 1
        except Exception as e:  # noqa: BLE001 —— 写盘失败要反馈给主线程
            self.error = e
            self.stop_evt.set()


# ===========================================================================
# 电平表
# ===========================================================================

def _db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-7))


class LevelMeter:
    """累计峰值/RMS/削波数。回调线程更新，主线程定期读取。

    回调里用非阻塞取锁：拿不到就跳过本次统计——电平表只是给人看的，
    绝不能因为统计而拖慢实时回调。
    """

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self._lock = threading.Lock()
        self._peak = np.zeros(channels)
        self._peak_hold = np.zeros(channels)
        self._sumsq = np.zeros(channels)
        self._count = 0
        self._clips = 0

    def update(self, block: np.ndarray) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            peak = np.max(np.abs(block), axis=0).astype(np.float64)
            np.maximum(self._peak, peak, out=self._peak)
            np.maximum(self._peak_hold, peak, out=self._peak_hold)
            self._sumsq += np.sum(np.square(block, dtype=np.float64), axis=0)
            self._count += block.shape[0]
            self._clips += int(np.count_nonzero(np.abs(block) >= 0.9995))
        finally:
            self._lock.release()

    def snapshot(self) -> dict:
        """取走本段时间窗的统计并清零（峰值保持量除外）。"""
        with self._lock:
            rms = np.sqrt(self._sumsq / max(self._count, 1))
            out = {
                "peak": self._peak.copy(),
                "peak_hold": self._peak_hold.copy(),
                "rms": rms,
                "clips": self._clips,
                "frames": self._count,
            }
            self._peak[:] = 0.0
            self._sumsq[:] = 0.0
            self._count = 0
            self._clips = 0
            return out

    def totals(self) -> dict:
        with self._lock:
            return {"peak_hold": self._peak_hold.copy(), "clips_total": self._clips}


# ===========================================================================
# 录制会话
# ===========================================================================

class Recording:
    """一次录音会话：建流、回调入队、写盘线程、收尾。"""

    def __init__(self, cfg: Config, device: int, out_path: str,
                 channels: int, extra_settings=None) -> None:
        self.cfg = cfg
        self.device = device
        self.out_path = out_path
        self.channels = channels
        self.samplerate = cfg.samplerate
        self.extra_settings = extra_settings

        self.stop_evt = threading.Event()
        self.q: queue.Queue = queue.Queue(
            maxsize=max(1, int(cfg.queue_seconds * cfg.samplerate
                              / max(cfg.blocksize or 1024, 1))))
        self.meter = LevelMeter(self.channels) if cfg.show_level else None

        # 监听：录音 + 实时转发到耳机共用同一个双工流
        self.monitor = cfg.monitor
        self.monitor_gain = float(10.0 ** (cfg.monitor_gain_db / 20.0))
        self.monitor_out_ch = cfg.monitor_out_channels
        self.mixer = None
        self.stream: sd.Stream | sd.InputStream | None = None

        self.dropped_blocks = 0
        self.status_flags: set[str] = set()
        self.writer: WavWriter | None = None

        self.t_start: float | None = None      # 起录的 monotonic
        self.t_start_unix: float | None = None  # 起录的墙钟（对齐视频用）
        self.t_start_local: str = ""

    # ---- 音频回调（实时线程，尽量短）----
    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.status_flags.add(str(status))
        if self.meter is not None:
            self.meter.update(indata)
        try:
            # 存盘的是**未加增益**的原始信号。监听增益只作用于耳机，
            # 若在这里放大后再存盘，录音电平会被监听设置悄悄改掉。
            self.q.put_nowait(indata.copy())
        except queue.Full:
            # 磁盘跟不上：丢这块并计数，绝不阻塞回调
            self.dropped_blocks += 1

    # ---- 双工回调：录音 + 监听 ----
    def _callback_duplex(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            self.status_flags.add(str(status))
        if self.meter is not None:
            self.meter.update(indata)
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            self.dropped_blocks += 1

        # 实时转发到耳机。映射交给 mixer：单路有信号时自动复制到两耳，
        # 否则按 1:1 走（话筒只接 Input 1 时右耳不会静音）。
        if self.mixer is not None:
            self.mixer.process(indata, outdata)
        else:
            # PortAudio 不保证 outdata 已清零；不写就会把旧缓冲反复送到耳机
            outdata.fill(0.0)

    def open(self, force: bool = False) -> None:
        extra = self.extra_settings

        # 预检：采样率/通道数不被驱动接受时，提前失败并给出人话
        if not force:
            try:
                sd.check_input_settings(
                    device=self.device, channels=self.channels,
                    samplerate=self.samplerate, dtype="float32",
                    extra_settings=extra)
                if self.monitor:
                    sd.check_output_settings(
                        device=self.device, channels=self.monitor_out_ch,
                        samplerate=self.samplerate, dtype="float32")
            except Exception as e:  # noqa: BLE001
                hint = ""
                try:
                    dinfo = sd.query_devices(self.device)
                    hint = (f"\n        该设备默认采样率 "
                            f"{dinfo['default_samplerate']:.0f} Hz，"
                            f"输入通道 {dinfo['max_input_channels']}")
                    if self.cfg.host_api.upper() == "ASIO":
                        hint += ("\n        ASIO 下采样率必须与 Focusrite Control 2 "
                                 "里设置的一致，改完重开程序。")
                except Exception:  # noqa: BLE001
                    pass
                raise SystemExit(f"[错误] 设备不接受当前设置: {e}{hint}\n"
                                 f"       （确知没问题可加 --force 跳过预检）")

        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

        self.writer = WavWriter(self.out_path, self.samplerate, self.channels,
                                self.q, self.stop_evt)
        self.writer.start()

        if self.monitor:
            # ASIO 独占：输入和输出必须在同一个流里，否则打不开
            dev = sd.query_devices(self.device)
            n_out = min(self.monitor_out_ch, int(dev["max_output_channels"]))
            if n_out < 1:
                raise SystemExit(
                    f"[错误] 要开监听，但设备 [{self.device}] {dev['name']} "
                    f"没有输出通道。耳机需插在这张声卡上。")
            if n_out != self.monitor_out_ch:
                print(f"[提示] 设备只有 {n_out} 个输出通道，"
                      f"监听按 {n_out} 通道走")
            self.monitor_out_ch = n_out
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
                extra_settings=extra,
                callback=self._callback_duplex,
            )
        else:
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                dtype="float32",
                blocksize=self.cfg.blocksize,
                extra_settings=extra,
                callback=self._callback,
            )

    def start(self) -> None:
        assert self.stream is not None
        self.t_start_unix = time.time()
        self.t_start_local = dt.datetime.fromtimestamp(
            self.t_start_unix).isoformat(timespec="milliseconds")
        self.stream.start()
        self.t_start = time.monotonic()

    def elapsed(self) -> float:
        if self.t_start is None:
            return 0.0
        return time.monotonic() - self.t_start

    def stop(self) -> None:
        """停止采集并等写盘线程把队列排空。"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
        self.stop_evt.set()
        try:
            self.q.put_nowait(None)          # 哨兵，让写盘线程立即收尾
        except queue.Full:
            pass
        if self.writer is not None:
            self.writer.join()

    # ---- 电平表一行 ----
    def level_line(self) -> str:
        assert self.meter is not None
        s = self.meter.snapshot()
        parts = []
        for ch in range(self.channels):
            parts.append(f"{_db(s['rms'][ch]):6.1f}")
        hold = _db(float(np.max(s["peak_hold"])))
        line = (f"[电平] RMS(dBFS) " + " ".join(parts)
                + f" | 峰值保持 {hold:6.1f} | {self.elapsed():7.1f}s")
        if s["clips"]:
            line += f"  ⚠ 削波 {s['clips']}"
        return line


# ===========================================================================
# 主流程
# ===========================================================================

def _setup_console_utf8() -> None:
    """Windows 控制台切到 UTF-8，否则中文输出会乱码。"""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:  # noqa: BLE001
            pass
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _wait_for_enter(stop_evt: threading.Event) -> None:
    """后台线程：读到一行（回车）就请求停止。

    注意 readline() 的两种返回必须区分开：
      - "\\n" 之类的非空串 = 用户按了回车 -> 停止录音；
      - "" = EOF，说明 stdin 是管道/文件，或已被关闭，**再也不会读到输入**。
        此时若照样停止，就会瞬间录出一个空文件（静默失败）。
        所以这里只发提示、不停止，让录音继续由 Ctrl+C / --duration 结束。

    这里刻意不用 sys.stdin.isatty() 判断：Git Bash 下 /dev/null 的
    isatty() 会返回 True，靠不住。
    """
    try:
        line = sys.stdin.readline()
    except Exception:  # noqa: BLE001
        return
    if line == "":
        print("\n[提示] 标准输入已结束，回车停止不可用；"
              "请用 Ctrl+C 或 --duration 停止。")
        return
    stop_evt.set()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从 Focusrite(ASIO) 录音并保存为 24-bit WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=CONFIG_FILE,
                   help="配置文件路径")
    p.add_argument("--save-config", default=None, metavar="PATH",
                   help="把当前生效配置写成模板后退出")
    p.add_argument("--list", action="store_true", help="列出音频设备后退出")
    p.add_argument("--out", default=None, help="保存根目录（覆盖配置 path）")
    p.add_argument("--name", default=None, help="文件名前缀（覆盖配置 baseName）")
    p.add_argument("--duration", type=float, default=None,
                   help="录制时长(秒)；不给则由回车/Ctrl+C 停止")
    p.add_argument("--device", default=None, help="设备名子串（覆盖配置 deviceName）")
    p.add_argument("--host-api", default=None, help="主机 API 名（覆盖配置 hostApi）")
    p.add_argument("--samplerate", type=int, default=None, help="采样率")
    p.add_argument("--channels", type=int, default=None, help="录制通道数")
    p.add_argument("--blocksize", type=int, default=None, help="回调块大小，0=自动")
    p.add_argument("--no-level", action="store_true", help="不显示电平表")
    p.add_argument("--monitor", action="store_true",
                   help="录音同时把输入转发到耳机（边录边听）")
    p.add_argument("--monitor-gain-db", type=float, default=None,
                   help="监听增益(dB)，只影响耳机音量")
    p.add_argument("--monitor-mode", choices=["auto", "stereo", "mono"],
                   default=None,
                   help="监听映射：auto=单路有信号时复制到两耳")
    p.add_argument("--force", action="store_true",
                   help="跳过开流前的 check_input_settings 预检")
    return p


def _unique_path(directory: str, base: str, stamp: str, ext: str) -> str:
    """避免同秒重名：必要时加 _1 / _2 后缀。"""
    cand = os.path.join(directory, f"{base}_{stamp}{ext}")
    n = 1
    while os.path.exists(cand):
        cand = os.path.join(directory, f"{base}_{stamp}_{n}{ext}")
        n += 1
    return cand


def main(argv=None) -> int:
    _setup_console_utf8()
    args = build_parser().parse_args(argv)

    cfg = Config()
    if load_config(cfg, args.config):
        print(f"[配置] 已载入 {args.config}")
    else:
        print(f"[配置] 未找到 {args.config}，使用内置默认值")
        if write_default_config(cfg, args.config):
            print(f"[配置] 已生成配置模板: {args.config}")

    # 命令行覆盖
    if args.out is not None:
        cfg.path = args.out
    if args.name is not None:
        cfg.base_name = args.name
    if args.duration is not None:
        cfg.duration = args.duration
    if args.device is not None:
        cfg.device_name = args.device
    if args.host_api is not None:
        cfg.host_api = args.host_api
    if args.samplerate is not None:
        cfg.samplerate = args.samplerate
    if args.channels is not None:
        cfg.channels = args.channels
    if args.blocksize is not None:
        cfg.blocksize = args.blocksize
    if args.no_level:
        cfg.show_level = False
    if args.monitor:
        cfg.monitor = True
    if args.monitor_gain_db is not None:
        cfg.monitor_gain_db = args.monitor_gain_db
    if args.monitor_mode is not None:
        cfg.monitor_mode = args.monitor_mode

    if args.save_config:
        ok = write_default_config(cfg, args.save_config)
        print(f"[配置] {'已写入 ' + args.save_config if ok else '写入失败'}")
        return 0 if ok else 1

    if args.list:
        print_devices()
        return 0

    print("===== 录音程序 =====")
    print_config(cfg)

    device, _api = resolve_device(cfg)
    info = sd.query_devices(device)
    channels, extra = resolve_channels(cfg, device)
    print(f"[设备] [{device}] {info['name']}  "
          f"(可录 {info['max_input_channels']} 通道, "
          f"默认 {info['default_samplerate']:.0f} Hz)")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = os.path.join(cfg.path, cfg.base_name)
    out_path = _unique_path(directory, cfg.base_name, stamp, ".wav")

    rec = Recording(cfg, device, out_path, channels, extra)
    try:
        rec.open(force=args.force)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[错误] 打开音频流失败: {e}")

    print(f"[录音] 写入 {out_path}")
    try:
        rec.start()
    except Exception as e:  # noqa: BLE001
        rec.stop()
        raise SystemExit(f"[错误] 启动音频流失败: {e}")

    if cfg.duration > 0:
        print(f"\n>>> 录制中（{cfg.duration:g} 秒后自动停止，Ctrl+C 提前结束）...\n")
    else:
        print("\n>>> 录制中。按回车停止...\n")
        # 回车停止的监听线程总是启动；stdin 若是管道/文件，
        # _wait_for_enter 会自己识别 EOF 并放弃（而不是误触发停止）。
        threading.Thread(target=_wait_for_enter, args=(rec.stop_evt,),
                         daemon=True).start()

    next_tick = time.monotonic()
    try:
        while not rec.stop_evt.wait(0.05):
            if cfg.duration > 0 and rec.elapsed() >= cfg.duration:
                rec.stop_evt.set()
                break
            if cfg.show_level and time.monotonic() >= next_tick:
                next_tick = time.monotonic() + cfg.level_interval
                sys.stdout.write("\r" + rec.level_line())
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[录音] 收到 Ctrl+C，停止...")
        rec.stop_evt.set()

    print("\n[录音] 停止中...")
    rec.stop()

    # 写盘线程自身出错（磁盘满 / 权限）
    if rec.writer is not None and rec.writer.error is not None:
        print(f"[错误] 写盘线程异常: {rec.writer.error}")

    frames = rec.writer.frames_written if rec.writer else 0
    duration = frames / cfg.samplerate if cfg.samplerate else 0.0

    # 旁车元数据：start_unix 用于与录像对齐
    meta = {
        "file": os.path.basename(out_path),
        "start_time_local": rec.t_start_local,
        "start_unix": rec.t_start_unix,
        "host_api": cfg.host_api,
        "device": info["name"],
        "samplerate": cfg.samplerate,
        "channels": rec.channels,
        "sampwidth_bytes": SAMPWIDTH,
        "bits_per_sample": SAMPWIDTH * 8,
        "duration_s": round(duration, 6),
        "frames": frames,
        "dropped_blocks": rec.dropped_blocks,
        "status_flags": sorted(rec.status_flags),
        "clip_samples": int(rec.meter.totals()["clips_total"]) if rec.meter else None,
        "config": {
            "input_channels": cfg.input_channels,
            "blocksize": cfg.blocksize,
        },
    }
    if frames > 0:
        json_path = os.path.splitext(out_path)[0] + ".json"
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[警告] 元数据写入失败: {e}")
    else:
        # 一帧都没有：删掉空 WAV，别在数据目录里留下无法区分的垃圾文件
        try:
            os.remove(out_path)
            print(f"[提示] 未采集到任何数据，已删除空文件 {out_path}")
        except OSError:
            pass

    print("----- 录音结果 -----")
    print(f"  文件      : {out_path}")
    print(f"  时长      : {duration:.3f} s  ({frames} 帧)")
    print(f"  采样率    : {cfg.samplerate} Hz, {rec.channels} ch, "
          f"{SAMPWIDTH*8}-bit")
    if rec.dropped_blocks:
        print(f"  ⚠ 丢弃块  : {rec.dropped_blocks}（磁盘写入跟不上）")
    if rec.status_flags:
        print(f"  ⚠ 状态    : {'; '.join(sorted(rec.status_flags))}")
    if rec.meter is not None:
        t = rec.meter.totals()
        print(f"  峰值保持  : {_db(float(np.max(t['peak_hold']))):.1f} dBFS")
        if t["clips_total"]:
            print(f"  ⚠ 削波样本: {t['clips_total']}（把 Focusrite 增益调小些）")
    print("--------------------")
    print("[主] 已退出。")
    return 0 if frames > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
