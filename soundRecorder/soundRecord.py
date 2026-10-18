# -*- coding: utf-8 -*-
"""实时波形监视（只看，不保存录音；要保存请用 record.py）。

    python soundRecord.py
    python soundRecord.py --list              # 看设备号和名字
    python soundRecord.py --device Focusrite  # 按名字选（推荐，序号会变）
    python soundRecord.py --device 10 --samplerate 48000

两个关键点
----------
1. ASIO 支持在 sounddevice 里默认关闭，必须在 import 之前设 SD_ENABLE_ASIO，
   否则 ASIO 设备根本不会出现在设备列表里。
2. matplotlib 只能在主线程里画。原版在音频回调线程里调 draw()，会随机崩溃
   或让窗口卡死。这里回调只负责把数据放进队列，绘图全在主线程做。
"""

import argparse
import os
import queue
import sys

import numpy as np

os.environ.setdefault("SD_ENABLE_ASIO", "1")  # 必须在 import sounddevice 之前

import matplotlib

import matplotlib.pyplot as plt  # noqa: E402
import sounddevice as sd  # noqa: E402

# 默认字体 DejaVu Sans 没有中文字形，轴标签会变成方框
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PLOT_FPS = 30  # 界面刷新率，与音频块大小无关


def list_devices():
    hostapis = sd.query_hostapis()
    for d in sd.query_devices():
        api = hostapis[d["hostapi"]]["name"]
        print(f"{d['index']:>3}  {api:<22} "
              f"in={d['max_input_channels']:<2} out={d['max_output_channels']:<2} "
              f"{d['name']}")


def pick_device(spec):
    """spec 可以是设备序号字符串、名字子串，或 None（系统默认输入）。"""
    if spec is None:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            sys.exit("没有默认输入设备，请用 --device 指定（先加 --list 看）")
        return int(idx)
    if spec.isdigit():
        idx = int(spec)
        if not 0 <= idx < len(sd.query_devices()):
            sys.exit(f"设备序号 {idx} 超出范围")
        if sd.query_devices(idx)["max_input_channels"] < 1:
            sys.exit(f"设备 {idx} 没有输入通道")
        return idx
    hits = [d for d in sd.query_devices()
            if spec.lower() in d["name"].lower() and d["max_input_channels"] > 0]
    if not hits:
        sys.exit(f"找不到名字含 '{spec}' 的输入设备（先加 --list 看）")
    if len(hits) > 1:
        print(f"'{spec}' 匹配到 {len(hits)} 个，用第一个: "
              f"[{hits[0]['index']}] {hits[0]['name']}")
    return hits[0]["index"]


def main():
    p = argparse.ArgumentParser(description="实时波形监视")
    p.add_argument("--list", action="store_true", help="列出设备后退出")
    p.add_argument("--device", default=None, help="设备序号或名字子串")
    p.add_argument("--samplerate", type=float, default=None,
                   help="默认用设备自身的默认采样率")
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--blocksize", type=int, default=1024)
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    device = pick_device(args.device)
    info = sd.query_devices(device)
    samplerate = int(args.samplerate or info["default_samplerate"])
    blocksize = args.blocksize

    # 回调 -> 主线程。只留最新几块，画不过来就丢旧的，绝不阻塞实时回调。
    q = queue.Queue(maxsize=4)
    dropped = [0]
    status_flags = set()

    def callback(indata, frames, time_info, status):
        if status:
            status_flags.add(str(status))
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            dropped[0] += 1

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.zeros(blocksize)
    (line,) = ax.plot(np.arange(blocksize), y, lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("样本")
    ax.set_ylabel("幅度")
    ax.grid(True, alpha=0.3)
    title = ax.set_title(f"{info['name']}")
    fig.tight_layout()

    print(f"[设备] [{device}] {info['name']}  "
          f"{samplerate} Hz, {args.channels} ch, 块 {blocksize}")
    print("关闭窗口或 Ctrl+C 退出。")

    stream = sd.InputStream(device=device, channels=args.channels,
                            samplerate=samplerate, blocksize=blocksize,
                            dtype="float32", callback=callback)
    stream.start()
    try:
        while plt.fignum_exists(fig.number):
            try:
                block = q.get(timeout=1.0 / PLOT_FPS)
            except queue.Empty:
                plt.pause(0.001)  # 没新数据也要让 GUI 处理事件，否则假死
                continue

            # 驱动不保证 frames == blocksize，补齐/截断到横轴长度
            n = min(len(block), blocksize)
            y[:] = 0.0
            y[:n] = block[:n]
            line.set_ydata(y)

            peak = float(np.max(np.abs(y)))
            db = 20 * np.log10(max(peak, 1e-7))
            title.set_text(f"{info['name']}  峰值 {db:6.1f} dBFS  "
                           f"丢块 {dropped[0]}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(1.0 / PLOT_FPS)
    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        stream.stop()
        stream.close()
        if dropped[0]:
            print(f"绘图跟不上，丢弃了 {dropped[0]} 块（不影响音频）")
        if status_flags:
            print("状态:", "; ".join(sorted(status_flags)))


if __name__ == "__main__":
    main()
