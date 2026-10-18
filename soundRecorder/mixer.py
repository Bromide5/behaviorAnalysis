# -*- coding: utf-8 -*-
"""
监听混音：把输入映射到耳机输出
============================

解决一个很常见的情况：**话筒只接在 Input 1，Input 2 是空的**。
此时若按 2 通道 1:1 映射，右耳拿到的就是静音，听起来像「耳机只有左边响」。

`MonitorMixer` 会自动识别这种情况：当某一路有信号、另一路只有底噪时，
把有信号的那一路复制到所有输出通道。检测带迟滞和驻留时间，避免在说话的
停顿处来回切换。也可以用 `mode` 强制指定行为：

    auto    自动（默认）：单路有信号 -> 复制到两耳；两路都有 -> 立体声
    stereo  始终 1:1 映射
    mono    始终把 Input 1 复制到所有输出

设计约束
--------
`process()` 在音频实时回调里被调用，必须**无分配、无锁、无 I/O**。
这里只做 NumPy 算术和几个标量比较。
"""

from __future__ import annotations

import numpy as np


class MonitorMixer:
    # 判定阈值：强于 -55 dBFS 算「有信号」，弱于 -70 dBFS 算「静音」。
    # 本机底噪约 -92 dBFS，留了 20 dB 余量，不会误判。
    ACTIVE_DB = -55.0
    SILENT_DB = -70.0
    DWELL_S = 0.30      # 候选状态需持续这么久才切换，防止抖动
    EMA_ALPHA = 0.90    # 电平平滑系数

    def __init__(self, in_ch: int, out_ch: int, gain: float = 1.0,
                 mode: str = "auto", samplerate: int = 48000) -> None:
        if mode not in ("auto", "stereo", "mono"):
            raise ValueError(f"未知的监听模式: {mode}")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.gain = float(gain)
        self.mode = mode
        self.samplerate = samplerate

        # 当前生效的映射
        self._cur = "mono" if (mode == "mono" or in_ch == 1) else "stereo"
        self._cur_src = 0
        self._ema = np.zeros(max(in_ch, 1))
        self._dwell = 0

    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        """当前映射，用于界面/日志显示。"""
        if self._cur == "mono":
            return f"单声道(输入{self._cur_src + 1}->两耳)"
        return "立体声"

    # ------------------------------------------------------------------
    def process(self, indata: np.ndarray, outdata: np.ndarray) -> None:
        """实时回调里调用。indata/outdata 形状为 (frames, channels)。"""
        outdata.fill(0.0)

        if self.mode == "auto" and indata.shape[1] >= 2:
            self._update_mode(indata)

        if self._cur == "mono":
            src = min(self._cur_src, indata.shape[1] - 1)
            outdata[:, :] = indata[:, src:src + 1] * self.gain
        else:
            k = min(indata.shape[1], outdata.shape[1])
            outdata[:, :k] = indata[:, :k] * self.gain

        np.clip(outdata, -1.0, 1.0, out=outdata)

    # ------------------------------------------------------------------
    def _update_mode(self, x: np.ndarray) -> None:
        power = np.mean(x * x, axis=0)
        self._ema = self.EMA_ALPHA * self._ema + (1.0 - self.EMA_ALPHA) * power
        lvl = 10.0 * np.log10(self._ema + 1e-14)

        l0, l1 = lvl[0], lvl[1]
        active0, active1 = l0 > self.ACTIVE_DB, l1 > self.ACTIVE_DB
        silent1, silent0 = l1 < self.SILENT_DB, l0 < self.SILENT_DB

        candidate = None
        if active0 and silent1:
            candidate = ("mono", 0)
        elif active1 and silent0:
            candidate = ("mono", 1)
        elif active0 and active1:
            candidate = ("stereo", self._cur_src)

        # 两路都静音时不作判断：保持当前映射，避免说话停顿时来回切换
        if candidate is None:
            self._dwell = 0
            return

        if candidate[0] == self._cur and (
                candidate[0] != "mono" or candidate[1] == self._cur_src):
            self._dwell = 0
            return

        self._dwell += x.shape[0]
        if self._dwell >= self.DWELL_S * self.samplerate:
            self._cur = candidate[0]
            if candidate[0] == "mono":
                self._cur_src = candidate[1]
            self._dwell = 0
