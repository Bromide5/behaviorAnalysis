# -*- coding: utf-8 -*-
"""
实时监听（把输入直接转发到耳机）
==============================

把声卡的输入实时送到输出，用来在耳机里听自己说话 / 听环境声。
可调增益，带实时电平表。

为什么必须用双工流
------------------
ASIO 设备是**独占**的：同一张声卡不能同时开一个输入流和一个输出流。
所以这里用 `sd.Stream`（一个流同时负责输入和输出），回调里把收到的
输入块直接抄进输出块。这也是低延迟的关键——数据不落地，不经过磁盘。

也因此：**本程序运行时 `record.py` 打不开同一张声卡**，反之亦然。
需要「边录边听」请用 `record.py --monitor`（见下）。

用法
----
    python monitor.py                          # 默认设备，0 dB
    python monitor.py --list                   # 看设备
    python monitor.py --device Focusrite       # 按名字选
    python monitor.py --gain-db 12             # 加 12 dB
    python monitor.py --blocksize 128          # 更低延迟
    python monitor.py --duration 10            # 10 秒后自动停

注意
----
- 用**耳机**监听。用音箱 + 话筒会形成声学正反馈，几秒内啸叫。
- 如果你已经在声卡上按下了硬件 DIRECT MONITOR 按钮，会听到两份声音
  （硬件一份、本程序一份），有轻微梳状滤波。二选一即可。
- 声卡上的输入增益旋钮决定录音电平；本程序的 --gain-db 只影响你**听到**
  的音量，不影响录音本身。录到的信号太小要拧声卡旋钮，而不是靠这里放大。
- 话筒只接 Input 1 时，2 通道 1:1 映射会让右耳静音。`--monitor-mode auto`
  （默认）会在检测到只有一路有信号时把它复制到两耳。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

# 必须在 import sounddevice 之前置位，否则 ASIO 设备根本不会出现在列表里
os.environ.setdefault("SD_ENABLE_ASIO", "1")

import sounddevice as sd  # noqa: E402

from mixer import MonitorMixer  # noqa: E402

METER_INTERVAL = 0.15  # 电平表刷新间隔（秒）


# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------

def list_devices() -> None:
    hostapis = sd.query_hostapis()
    print("===== 音频设备 =====")
    for d in sd.query_devices():
        print(f"  {d['index']:>3}  {hostapis[d['hostapi']]['name']:<20} "
              f"in={d['max_input_channels']:<2} "
              f"out={d['max_output_channels']:<2} {d['name']}")
    print("\n实时监听需要同时有输入和输出通道的设备。")


def _apps() -> list[dict]:
    hostapis = sd.query_hostapis()
    return [dict(d, api=hostapis[d["hostapi"]]["name"])
            for d in sd.query_devices()]


def pick_device(spec: str | None) -> int:
    """解析设备：序号字符串 / 名字子串 / None（自动挑一个能做双工的设备）。"""
    apps = _apps()

    if spec is None:
        # 不能用 sd.default.device[0]：那通常指向 MME 的纯输入端点。
        # MME/WASAPI 下 Focusrite 的输入、输出是两个独立端点，各自都没有
        # 双工能力，因此必须挑一个 in>0 且 out>0 的设备——实际只有 ASIO 满足。
        duplex = [d for d in apps
                  if d["max_input_channels"] > 0 and d["max_output_channels"] > 0]
        if duplex:
            duplex.sort(key=lambda d: d["api"] != "ASIO")  # ASIO 优先
            d = duplex[0]
            print(f"[自动] 选用可双工的设备 [{d['index']}] "
                  f"{d['api']} / {d['name']}")
            return int(d["index"])
        sys.exit("[错误] 没有找到同时具备输入和输出的设备。\n"
                 "       实时监听需要在同一个设备上同时收发；"
                 "请加 --list 查看，并用 --device 指定。")

    if spec.isdigit():
        idx = int(spec)
        if not 0 <= idx < len(apps):
            sys.exit(f"设备序号 {idx} 超出范围")
        return idx

    hits = [d for d in apps if spec.lower() in d["name"].lower()]
    if not hits:
        sys.exit(f"找不到名字含 '{spec}' 的设备（先加 --list 看）")
    if len(hits) > 1:
        # 同名设备在不同主机 API 下会重复出现，优先选能双工的
        duplex = [d for d in hits
                  if d["max_input_channels"] > 0 and d["max_output_channels"] > 0]
        if duplex:
            duplex.sort(key=lambda d: d["api"] != "ASIO")
            hits = duplex
            print(f"'{spec}' 匹配到多个，选用可双工的: "
                  f"[{hits[0]['index']}] {hits[0]['api']} / {hits[0]['name']}")
        else:
            print(f"'{spec}' 匹配到 {len(hits)} 个（都不支持双工），用第一个: "
                  f"[{hits[0]['index']}] {hits[0]['api']} / {hits[0]['name']}")
    return int(hits[0]["index"])


def check_duplex(device: int) -> None:
    d = sd.query_devices(device)
    api = sd.query_hostapis()[d["hostapi"]]["name"]
    if d["max_input_channels"] < 1:
        sys.exit(f"[错误] [{device}] {api} / {d['name']} 没有输入通道，无法监听。")
    if d["max_output_channels"] < 1:
        # MME / DirectSound / WASAPI 下，声卡的输入和输出被拆成两个独立端点，
        # 每个端点只具备单向能力，因此这几个主机 API 都做不了双工。
        sys.exit(
            f"[错误] [{device}] {api} / {d['name']} 没有输出通道，无法监听。\n"
            f"\n"
            f"       在 {api} 下，Focusrite 的输入和输出是两个独立端点，\n"
            f"       任何一个都不能同时收发。实时监听要用 ASIO（它才是双工的）。\n"
            f"\n"
            f"       不加 --device 时程序会自动选；要手动指定就填 ASIO 那一条：\n"
            f"         python monitor.py --device Focusrite\n"
            f"       先跑 --list 确认哪一行标着 ASIO。")


# ---------------------------------------------------------------------------
# 电平
# ---------------------------------------------------------------------------

def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


class Meter:
    """回调线程写，主线程读。拿不到锁就跳过本次统计，绝不阻塞实时回调。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peak = 0.0
        self._sumsq = 0.0
        self._count = 0
        self._clips = 0

    def update(self, block: np.ndarray) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._peak = max(self._peak, float(np.max(np.abs(block))))
            self._sumsq += float(np.sum(np.square(block, dtype=np.float64)))
            self._count += block.size
            self._clips += int(np.count_nonzero(np.abs(block) >= 0.9995))
        finally:
            self._lock.release()

    def take(self) -> tuple[float, float, int]:
        with self._lock:
            peak = self._peak
            rms = np.sqrt(self._sumsq / max(self._count, 1))
            clips = self._clips
            self._peak, self._sumsq, self._count, self._clips = 0.0, 0.0, 0, 0
            return peak, float(rms), clips


# ---------------------------------------------------------------------------
# 双工转发
# ---------------------------------------------------------------------------

class Passthrough:
    def __init__(self, device: int, samplerate: int, in_ch: int, out_ch: int,
                 blocksize: int, gain_db: float, mode: str = "auto") -> None:
        self.device = device
        self.samplerate = samplerate
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.blocksize = blocksize
        self.gain = float(10.0 ** (gain_db / 20.0))
        self.mode = mode
        self.meter = Meter()
        self.status_flags: set[str] = set()
        self.stream: sd.Stream | None = None
        self.mixer: MonitorMixer | None = None

    def callback(self, indata, outdata, frames, time_info, status) -> None:
        """实时线程。只做一次乘法和拷贝，越短越好。"""
        if status:
            self.status_flags.add(str(status))

        # 映射交给 mixer：单路有信号时复制到两耳，两路都有信号时按 1:1 走
        if self.mixer is not None:
            self.mixer.process(indata, outdata)
        else:
            outdata.fill(0.0)

        # 电平取**送去耳机**的信号，这才是你实际听到的
        self.meter.update(outdata)

    def run(self, duration: float) -> int:
        check_duplex(self.device)
        dev = sd.query_devices(self.device)

        # 输入通道不能超过设备实际能提供的最多路数
        in_ch = min(self.in_ch, int(dev["max_input_channels"]))
        out_ch = min(self.out_ch, int(dev["max_output_channels"]))
        if in_ch != self.in_ch or out_ch != self.out_ch:
            print(f"[提示] 设备只有 {dev['max_input_channels']} 入 / "
                  f"{dev['max_output_channels']} 出，"
                  f"实际使用 {in_ch} 入 / {out_ch} 出")
        self.in_ch, self.out_ch = in_ch, out_ch

        self.mixer = MonitorMixer(
            in_ch=in_ch, out_ch=out_ch, gain=self.gain,
            mode=self.mode, samplerate=self.samplerate)

        print(f"[设备] [{self.device}] {dev['name']}")
        print(f"[参数] {self.samplerate} Hz, {in_ch} 入 -> {out_ch} 出, "
              f"块 {self.blocksize}, 增益 {db(self.gain):+.1f} dB, "
              f"监听模式 {self.mode}")

        # 先预检，采样率不被接受时给出人话而不是底层异常
        try:
            sd.check_input_settings(device=self.device, channels=in_ch,
                                    samplerate=self.samplerate, dtype="float32")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[错误] 输入不接受当前设置: {e}\n"
                     f"       设备默认采样率 {dev['default_samplerate']:.0f} Hz。"
                     f"ASIO 下要与 Focusrite 控制里的设置一致。")
        try:
            sd.check_output_settings(device=self.device, channels=out_ch,
                                     samplerate=self.samplerate, dtype="float32")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[错误] 输出不接受当前设置: {e}")

        self.stream = sd.Stream(
            device=self.device,
            channels=(in_ch, out_ch),
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="float32",
            latency="low",
            callback=self.callback,
        )
        self.stream.start()

        print("\n>>> 监听中。Ctrl+C 停止。\n")
        print("[提示] 用耳机听。用音箱+话筒会啸叫。"
              "声卡上若已按 DIRECT MONITOR，请关掉（否则听到两份）。\n")

        t0 = time.monotonic()
        next_tick = t0
        try:
            while True:
                time.sleep(0.05)
                now = time.monotonic()
                if duration > 0 and now - t0 >= duration:
                    break
                if now >= next_tick:
                    next_tick = now + METER_INTERVAL
                    peak, rms, clips = self.meter.take()
                    line = (f"\r[电平] 峰值 {db(peak):6.1f} dBFS  "
                            f"RMS {db(rms):6.1f} dBFS  "
                            f"{now - t0:6.1f}s")
                    if clips:
                        line += f"  ⚠ 削波 {clips}"
                    # 信号太小时给出可操作的提示，而不是让人对着 -95 dBFS 发呆
                    if peak < 1e-3 and now - t0 > 1.0:
                        line += "  ← 几乎无信号: 拧声卡输入增益 / 检查话筒接线"
                    sys.stdout.write(line)
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n[监听] 收到 Ctrl+C，停止。")
        finally:
            self.stop()

        # 真实延迟：块周期 * 2 + 驱动缓冲。PortAudio 对 ASIO 报的 latency
        # 是固定占位值（本机恒为 47.9/49.9 ms），与 blocksize 无关，不可信。
        block_ms = self.blocksize / self.samplerate * 1000
        print(f"\n[延迟估算] 块周期 {block_ms:.1f} ms × 2 = 约 {block_ms*2:.0f} ms"
              f"（不含驱动内部缓冲；人耳对 >20 ms 会觉察到回声感）")
        return 0

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
        if self.status_flags:
            print(f"[状态] {'; '.join(sorted(self.status_flags))}")


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="把声卡输入实时转发到耳机",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--list", action="store_true", help="列出设备后退出")
    p.add_argument("--device", default=None,
                   help="设备序号或名字子串；默认用系统默认输入设备")
    p.add_argument("--samplerate", type=int, default=None,
                   help="采样率；默认用设备自身的默认值")
    p.add_argument("--in-channels", type=int, default=None,
                   help="输入通道数；默认等于设备可用的最小值(1)")
    p.add_argument("--out-channels", type=int, default=2, help="输出通道数")
    p.add_argument("--blocksize", type=int, default=256,
                   help="回调块大小；越小延迟越低，但更容易爆音")
    p.add_argument("--gain-db", type=float, default=0.0,
                   help="监听增益(dB)。只影响你听到的音量，不影响录音")
    p.add_argument("--monitor-mode", choices=("auto", "stereo", "mono"),
                   default="auto",
                   help="auto=单路有信号时复制到两耳；stereo=严格 1:1；"
                        "mono=输入1 复制到所有输出")
    p.add_argument("--duration", type=float, default=0.0,
                   help="运行秒数；0 表示一直运行到 Ctrl+C")
    return p


def main(argv=None) -> int:
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args(argv)
    if args.list:
        list_devices()
        return 0

    if args.blocksize < 16:
        sys.exit("[错误] --blocksize 至少 16")

    device = pick_device(args.device)
    dev = sd.query_devices(device)
    samplerate = args.samplerate or int(dev["default_samplerate"])
    # 默认只取 1 路输入：单声道监听更稳定，也更省带宽。
    # 双声道输入时可用 --in-channels 2。
    in_ch = args.in_channels or 1

    passthrough = Passthrough(device, samplerate, in_ch, args.out_channels,
                              args.blocksize, args.gain_db, args.monitor_mode)
    try:
        return passthrough.run(args.duration)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[错误] {e}")


if __name__ == "__main__":
    sys.exit(main())
