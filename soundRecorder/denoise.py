# -*- coding: utf-8 -*-
"""
离线降噪：高通 / 工频陷波 / 谱减
================================

为什么不做实时
--------------
谱减需要 STFT（1024 点，约 21 ms）和逐帧幅度运算。放进音频回调会引入
额外延迟，而且实时估计噪声谱很不稳定，容易出现「音乐噪声」。所以这里
做**离线后处理**：原始 WAV 原样保留，结果另存为 `_clean.wav`。

算法
----
1. **高通 1 kHz**（默认）：切掉低频隆隆、交流声和目标信号以下的噪声。
   采集链路如果已经做了同样的高通，可以用 `--highpass 0` 关掉，避免叠加。
2. **低通 20 kHz**（默认）：切掉超声/高频嘶声；48 kHz 采样下只影响
   20–24 kHz，基本是保险措施。
3. **工频陷波**（默认关闭）：高通 1 kHz 已经覆盖 50/100 Hz，只有在高通
   更低或工频峰高于 1 kHz 时才需要。
4. **谱减**：从录音里最安静的若干 1 秒窗口估计噪声谱，对每个 STFT
   帧做 `|X| - over * |N|`，并保留谱地板，避免把信号一并削掉。

用本机录音实测（48 kHz、24-bit）：`over=2.0, floor=-15 dB` 时安静段
RMS 从 -45.7 降到 -59.1 dBFS（约 13.5 dB），最响段只降 0.3 dB。
`over` 超过 2.5 会开始出现流水声/鸟叫似的伪影。

用法
----
    python denoise.py rec/rec_xxx.wav          # 输出 rec/rec_xxx_clean.wav
    python denoise.py rec                      # 处理目录下所有 wav
    python denoise.py rec/rec_xxx.wav --over 2.5 --floor-db -20
    python denoise.py rec/rec_xxx.wav --no-spectral     # 只做滤波
    python denoise.py rec/rec_xxx.wav --highpass 0 --lowpass 0   # 只做谱减
    python denoise.py rec/rec_xxx.wav --noise-file noise.wav   # 指定噪声样本

输出
----
- `<名字>_clean.wav`：24-bit PCM，采样率/声道数与原文件一致
- `<名字>_clean.json`：参数、降噪前后 RMS、噪声估计窗口位置
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import wave

import numpy as np
from scipy import signal

NFFT = 1024
HOP = NFFT // 4
QUIET_WINDOW_S = 1.0


# ---------------------------------------------------------------------------
# WAV 读写（24-bit PCM，和 record.py 保持一致）
# ---------------------------------------------------------------------------

def wav_info(path: str) -> tuple[int, int, int, int]:
    with wave.open(path, "rb") as w:
        if w.getcomptype() != "NONE":
            raise ValueError(f"{os.path.basename(path)}: 只支持未压缩 PCM WAV")
        return (w.getnchannels(), w.getsampwidth(),
                w.getframerate(), w.getnframes())


def decode(raw: bytes, sampwidth: int, channels: int) -> np.ndarray:
    """PCM 字节 -> float64，形状 (frames, channels)。"""
    if sampwidth == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 3:
        u = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (u[:, 0].astype(np.int32)
             | (u[:, 1].astype(np.int32) << 8)
             | (u[:, 2].astype(np.int32) << 16))
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        x = v.astype(np.float64) / float(1 << 23)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"不支持的位深: {sampwidth * 8} bit")
    return x.reshape(-1, channels)


def float_to_pcm24(x: np.ndarray) -> bytes:
    y = np.clip(x, -1.0, 1.0)
    i = np.rint(y * 8388608.0)
    i = np.clip(i, -(1 << 23), (1 << 23) - 1)
    # scipy.sosfilt 会返回 Fortran 序数组；view 要求最后一维连续
    i = np.ascontiguousarray(i.astype("<i4", order="C"))
    return i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def iter_blocks(path: str, block_frames: int):
    ch, sw, sr, n = wav_info(path)
    with wave.open(path, "rb") as w:
        while True:
            raw = w.readframes(block_frames)
            if not raw:
                break
            yield decode(raw, sw, ch)


# ---------------------------------------------------------------------------
# 滤波器
# ---------------------------------------------------------------------------

def build_sos(sr: int, highpass: float, lowpass: float,
              notches: list[float]) -> list[np.ndarray]:
    sos_list = []
    if highpass and highpass > 0:
        # 4 阶巴特沃斯，截止频率处约 -3 dB
        sos_list.append(signal.butter(4, highpass, "highpass",
                                      fs=sr, output="sos"))
    if lowpass and 0 < lowpass < sr / 2:
        sos_list.append(signal.butter(4, lowpass, "lowpass",
                                      fs=sr, output="sos"))
    for f0 in notches:
        if 0 < f0 < sr / 2:
            b, a = signal.iirnotch(f0, 30.0, fs=sr)
            sos_list.append(signal.tf2sos(b, a))
    return sos_list


class FilterBank:
    """逐块滤波并保留状态，避免块边界出现咔哒声。"""

    def __init__(self, sos_list: list[np.ndarray], channels: int) -> None:
        self.sos_list = sos_list
        self.zis = [np.zeros((s.shape[0], 2, channels)) for s in sos_list]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x
        for i, sos in enumerate(self.sos_list):
            y, self.zis[i] = signal.sosfilt(sos, y, axis=0, zi=self.zis[i])
        return y


# ---------------------------------------------------------------------------
# 噪声谱估计：找最安静的 1 秒窗口
# ---------------------------------------------------------------------------

def find_quiet_windows(path: str, channels: int, sr: int, count: int
                       ) -> list[list[tuple[float, int, int]]]:
    """每个声道各保留 count 个 RMS 最低的窗口：[(rms, start, nframes), ...]。"""
    win = max(int(sr * QUIET_WINDOW_S), NFFT)
    best: list[list[tuple[float, int, int]]] = [[] for _ in range(channels)]
    pos = 0
    with wave.open(path, "rb") as w:
        ch, sw, _, _ = wav_info(path)
        while True:
            raw = w.readframes(win)
            if not raw:
                break
            x = decode(raw, sw, ch)
            if len(x) < NFFT:
                break
            rms = np.sqrt(np.mean(x * x, axis=0))
            for c in range(channels):
                item = (float(rms[c]), pos, len(x))
                if len(best[c]) < count:
                    best[c].append(item)
                    best[c].sort(key=lambda t: -t[0])
                elif item[0] < best[c][0][0]:
                    best[c][0] = item
                    best[c].sort(key=lambda t: -t[0])
            pos += len(x)
    for c in range(channels):
        best[c].sort(key=lambda t: -t[0])  # 由大到小，便于日志显示
    return best


def _window(win: np.ndarray) -> np.ndarray:
    return np.sqrt(np.hanning(len(win)))


def read_window(path: str, start: int, n: int) -> np.ndarray:
    ch, sw, _, _ = wav_info(path)
    with wave.open(path, "rb") as w:
        w.setpos(start)
        raw = w.readframes(n)
    return decode(raw, sw, ch)


# ---------------------------------------------------------------------------
# 流式 STFT 谱减 + 重叠相加
# ---------------------------------------------------------------------------

class SpectralSubtractor:
    """块式 STFT 谱减。输入块长度必须是 HOP 的整数倍。

    OLA 会引入 `NFFT - HOP` 个样本的预滚延迟，驱动部分负责丢弃。
    """

    def __init__(self, noise_mag: np.ndarray, channels: int,
                 over: float = 2.0, floor_db: float = -15.0,
                 nfft: int = NFFT, hop: int = HOP) -> None:
        self.noise = noise_mag
        self.over = float(over)
        self.floor = float(10.0 ** (floor_db / 20.0))
        self.nfft = nfft
        self.hop = hop
        self.win = _window(np.zeros(nfft))
        self.tail = np.zeros((nfft - hop, channels))
        self.acc = np.zeros((nfft - hop, channels))
        self.norm = np.zeros(nfft - hop)

    def process(self, x: np.ndarray) -> np.ndarray:
        nfft, hop = self.nfft, self.hop
        buf = np.concatenate([self.tail, x], axis=0)
        L = len(buf)
        nframes = 1 + (L - nfft) // hop

        idx = (np.arange(nframes)[:, None] * hop
               + np.arange(nfft)[None, :])
        frames = buf[idx] * self.win[:, None]
        S = np.fft.rfft(frames, axis=1)
        mag = np.abs(S)

        sub = np.maximum(mag - self.over * self.noise[None, :, :], 0.0)
        sub = np.maximum(sub, self.floor * mag)   # 谱地板，压制音乐噪声
        Y = sub * np.exp(1j * np.angle(S))
        y = np.fft.irfft(Y, n=nfft, axis=1) * self.win[:, None]

        out = np.concatenate([self.acc, np.zeros((L, buf.shape[1]))], axis=0)
        nrm = np.concatenate([self.norm, np.zeros(L)])
        w2 = self.win ** 2
        for k in range(nframes):
            out[k * hop:k * hop + nfft] += y[k]
            nrm[k * hop:k * hop + nfft] += w2

        emit_n = nframes * hop
        self.tail = buf[-(nfft - hop):].copy()
        self.acc = out[emit_n:].copy()
        self.norm = nrm[emit_n:].copy()
        return out[:emit_n] / np.maximum(nrm[:emit_n, None], 1e-8)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def output_paths(path: str, out_dir: str | None, overwrite: bool) -> tuple[str, str]:
    d = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    wav = os.path.join(d, f"{stem}_clean.wav")
    js = os.path.join(d, f"{stem}_clean.json")
    if not overwrite and os.path.exists(wav):
        raise FileExistsError(wav)
    return wav, js


def denoise_file(path: str, args) -> dict | None:
    t0 = time.monotonic()
    ch, sw, sr, n = wav_info(path)
    if sw != 3:
        print(f"[提示] {os.path.basename(path)} 是 {sw * 8}-bit，"
              f"输出统一为 24-bit")

    try:
        out_wav, out_json = output_paths(path, args.out_dir, args.overwrite)
    except FileExistsError as e:
        print(f"[跳过] 目标已存在（加 --overwrite 覆盖）: {e}")
        return None

    if args.lowpass and args.lowpass >= sr / 2:
        print(f"[提示] 低通 {args.lowpass:g} Hz 不低于 Nyquist "
              f"{sr / 2:g} Hz，已忽略")
    sos_list = build_sos(sr, args.highpass, args.lowpass, args.notches)
    hop, nfft = HOP, NFFT
    block_frames = max(hop, int(args.chunk_seconds * sr) // hop * hop)

    # ---- 噪声谱 ----
    noise_mag = None
    noise_rms = None
    quiet_meta: list[list[dict]] = []
    win = _window(np.zeros(nfft))

    if args.spectral and args.noise_file:
        nch, _nsw, nsr, _ = wav_info(args.noise_file)
        if nsr != sr or nch != ch:
            raise ValueError(
                f"噪声文件需为 {sr} Hz / {ch} 声道，实际 {nsr} Hz / {nch} 声道")
        nf = FilterBank(sos_list, ch)
        accs, cnt = None, 0
        for b in iter_blocks(args.noise_file, block_frames):
            b = nf(b)
            if len(b) < nfft:
                continue
            fr = np.lib.stride_tricks.sliding_window_view(
                b, nfft, axis=0)[::hop]
            s = np.abs(np.fft.rfft(fr * win[:, None], axis=1)).sum(axis=0)
            accs = s if accs is None else accs + s
            cnt += len(fr)
        noise_mag = accs / max(cnt, 1)
        print(f"[噪声] 使用噪声文件 {os.path.basename(args.noise_file)}"
              f"（{cnt} 帧）")

    elif args.spectral:
        best = find_quiet_windows(path, ch, sr, args.quiet_windows)
        for items in best:
            quiet_meta.append([
                {"start_s": round(s / sr, 2), "rms_dbfs": round(_db(r), 1)}
                for r, s, _n in items])
        # 每个声道用自己最安静的那批窗口估计噪声谱
        total = None
        for c in range(ch):
            nf = FilterBank(sos_list, 1)
            acc, cnt = None, 0
            for _r, start, length in best[c]:
                x = nf(read_window(path, start, length)[:, c:c + 1])[:, 0]
                if len(x) < nfft:
                    continue
                fr = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop]
                s = np.abs(np.fft.rfft(fr * win, axis=1)).sum(axis=0)
                acc = s if acc is None else acc + s
                cnt += len(fr)
            col = (acc / max(cnt, 1)) if acc is not None else np.zeros(nfft // 2 + 1)
            col = col[:, None]
            total = col if total is None else np.hstack([total, col])
        noise_mag = total
        if best and best[0]:
            noise_rms = float(np.mean([r for r, _s, _n in best[0]]))
            print(f"[噪声] 取 {args.quiet_windows} 个最安静的 1 秒窗口，"
                  f"ch0 底噪约 {_db(noise_rms):.1f} dBFS")

    if noise_mag is None:
        noise_mag = np.zeros((nfft // 2 + 1, ch))

    # ---- 整体电平：用于报告和安全性提示 ----
    total_sumsq = 0.0
    total_cnt = 0
    for b in iter_blocks(path, block_frames):
        total_sumsq += float(np.sum(b * b))
        total_cnt += b.size
    overall_rms = float(np.sqrt(total_sumsq / max(total_cnt, 1)))

    den = SpectralSubtractor(noise_mag, ch, args.over, args.floor_db) \
        if args.spectral else None
    fb = FilterBank(sos_list, ch)
    # STFT 的重叠相加要预滚 NFFT-HOP 个样本；纯滤波没有这个延迟
    skip_left = (nfft - hop) if den is not None else 0
    stats = {"sumsq": 0.0, "count": 0, "peak": 0.0}
    written = 0

    with wave.open(out_wav, "wb") as outw:
        outw.setnchannels(ch)
        outw.setsampwidth(3)
        outw.setframerate(sr)

        def write_y(y: np.ndarray) -> None:
            nonlocal written, skip_left
            if skip_left:
                k = min(skip_left, len(y))
                y = y[k:]
                skip_left -= k
            if written >= n or len(y) == 0:
                return
            y = y[:n - written]
            outw.writeframes(float_to_pcm24(y))
            stats["sumsq"] += float(np.sum(y * y))
            stats["count"] += y.size
            stats["peak"] = max(stats["peak"], float(np.max(np.abs(y))))
            written += len(y)

        for b in iter_blocks(path, block_frames):
            x = fb(b)
            if den is not None:
                # 谱减要求块长是 HOP 的整数倍，末块补零（超出部分会被丢掉）
                if len(x) % hop:
                    x = np.vstack([x, np.zeros((hop - len(x) % hop, ch))])
                x = den.process(x)
            write_y(x)

        # 冲掉 OLA 尾部：喂零块，直到所有真实样本都已写出
        if den is not None:
            zeros = np.zeros((block_frames, ch))
            while written < n:
                write_y(den.process(zeros))

    out_rms = float(np.sqrt(stats["sumsq"] / max(stats["count"], 1)))
    out_peak = stats["peak"]

    meta = {
        "input": os.path.abspath(path),
        "output": os.path.abspath(out_wav),
        "samplerate": sr,
        "channels": ch,
        "frames": n,
        "duration_s": round(n / sr, 2),
        "params": {
            "spectral": bool(args.spectral),
            "over": args.over,
            "floor_db": args.floor_db,
            "highpass_hz": args.highpass,
            "lowpass_hz": args.lowpass,
            "notches_hz": args.notches,
            "quiet_windows": args.quiet_windows,
            "noise_file": os.path.abspath(args.noise_file) if args.noise_file else None,
        },
        "input_rms_dbfs": round(_db(overall_rms), 2),
        "output_rms_dbfs": round(_db(out_rms), 2),
        "output_peak_dbfs": round(_db(out_peak), 2),
        "clipped": bool(out_peak > 1.0),
        "noise_rms_dbfs_ch0": round(_db(noise_rms), 2) if noise_rms else None,
        "quiet_windows": quiet_meta,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[完成] {os.path.basename(out_wav)}  "
          f"{n / sr:.1f}s  RMS {_db(overall_rms):.1f} -> {_db(out_rms):.1f} dBFS  "
          f"峰值 {_db(out_peak):.1f} dBFS  ({meta['elapsed_s']:.1f}s)")
    if out_peak > 1.0:
        print("[警告] 处理后浮点峰值超过 0 dBFS（滤波器在瞬态上有过冲），"
              "已按 24-bit 上限截幅。输入本身若已接近满刻度，先降低录音增益。")
    if noise_rms and _db(noise_rms) > _db(overall_rms) - 10:
        print("[警告] 安静段和整体电平只差不到 10 dB，说明全程都有较强背景声；"
              "谱减可能连目标信号一起削弱，建议先 --no-spectral 只做滤波，"
              "或用 --noise-file 指定纯噪声样本。")
    return meta


def _db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


# ---------------------------------------------------------------------------

def collect_inputs(spec: str) -> list[str]:
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "*.wav")))
    elif any(c in spec for c in "*?["):
        files = sorted(glob.glob(spec))
    else:
        files = [spec]
    # 跳过已经处理过的结果，避免反复降噪
    return [f for f in files if not f.endswith("_clean.wav")]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="离线降噪：高通 / 低通（默认 1k–20k）+ 谱减（原始文件不动）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="?", default="rec",
                   help="WAV 文件、目录或通配符")
    p.add_argument("--out-dir", default=None, help="输出目录；默认与原文件同目录")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的 _clean.wav")
    p.add_argument("--spectral", action=argparse.BooleanOptionalAction,
                   default=True, help="是否做谱减")
    p.add_argument("--over", type=float, default=2.0,
                   help="谱减过减系数；越大越干净，>2.5 易出伪影")
    p.add_argument("--floor-db", type=float, default=-15.0,
                   help="谱地板(dB)；越低越干净，伪影越多")
    p.add_argument("--highpass", type=float, default=1000.0,
                   help="高通截止频率(Hz)；0 = 关闭")
    p.add_argument("--lowpass", type=float, default=20000.0,
                   help="低通截止频率(Hz)；0 = 关闭")
    p.add_argument("--notch", default="0",
                   help="工频陷波频率，逗号分隔；0 = 关闭（高通 1 kHz 已覆盖工频）")
    p.add_argument("--noise-file", default=None,
                   help="用这个 WAV 作为纯噪声样本估计噪声谱")
    p.add_argument("--quiet-windows", type=int, default=16,
                   help="用于估计噪声谱的最安静 1 秒窗口数")
    p.add_argument("--chunk-seconds", type=float, default=10.0,
                   help="分块处理的块长（秒），只影响内存占用")
    return p


def main(argv=None) -> int:
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args(argv)
    args.notches = [float(s) for s in str(args.notch).split(",")
                    if s.strip() and float(s) > 0]
    if args.quiet_windows < 1:
        sys.exit("[错误] --quiet-windows 至少 1")
    if not 0.0 <= args.over <= 4.0:
        sys.exit("[错误] --over 需在 0 ~ 4（0 = 不减，用于测试）")

    files = collect_inputs(args.input)
    if not files:
        sys.exit(f"[错误] 没有找到 wav: {args.input}")

    ok = 0
    for path in files:
        try:
            if denoise_file(path, args) is not None:
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[错误] {os.path.basename(path)}: {e}")
    print(f"\n[汇总] 处理 {ok}/{len(files)} 个文件")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
