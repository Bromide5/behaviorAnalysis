#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROI 交互标定工具 —— 给 analysis.py 手动框选社交区 / 非社交区。

用法：
    python roi_tool.py --video rec/trial1.avi --out roi_config.json
    python roi_tool.py --video frames/ --fps 30 --out roi_config.json
    python roi_tool.py --video rec/trial1.avi --config roi_config.json  # 载入已有配置再微调

交互：
    鼠标左键拖拽          在当前模式下画一个矩形
    1 / 2                 切换当前编辑的区域：1=社交区(绿) 2=非社交区(红)
    滑块 或 a / d         前后翻帧，挑一张清晰的来标
    u                     清除当前区域
    r                     清除全部
    s                     保存到 --out 指定的 JSON
    q / ESC               退出（有未保存修改会提示）

坐标一律基于**原始图像像素**（左上为原点，与 analysis.py 的 --social /
--nonsocial 完全一致），因此与 --crop / 预处理无关，可直接复用。

输出 JSON 可直接喂给分析程序：
    python analysis.py --video rec/trial1.avi --config roi_config.json --out results/
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

import analysis          # 复用其帧源（视频 / 图像序列 / 16 位 TIFF 处理）

WINDOW = "ROI Tool"

# 每种 ROI 的显示颜色（BGR）
COLORS = {
    "social":    (0, 200, 0),      # 绿
    "nonsocial": (0, 0, 220),      # 红
}
MODE_KEYS = {"1": "social", "2": "nonsocial"}
MODE_HINT = {"social": "社交区 (刺激鱼一侧)", "nonsocial": "非社交区 (空白一侧)"}


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def screen_size() -> tuple[int, int]:
    """屏幕分辨率；取不到时给个保守默认。"""
    try:
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def fit_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    """把 (w,h) 缩放到不超过 (max_w,max_h)，不放大。"""
    return min(1.0, max_w / w, max_h / h)


def normalize_rect(a, b) -> tuple[int, int, int, int]:
    x1, y1 = a
    x2, y2 = b
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)


# --------------------------------------------------------------------------
# 标定器
# --------------------------------------------------------------------------


class RoiPicker:
    def __init__(self, src, out_path: str, cfg: dict):
        self.src = src
        self.out_path = out_path
        self.cfg = cfg                       # 原样保留其它键，仅回写 ROI

        self.mode = "social"
        self.rois: dict[str, tuple[int, int, int, int] | None] = {
            "social": None, "nonsocial": None,
        }
        # 从已有配置里恢复
        for k in ("social", "nonsocial"):
            v = cfg.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                self.rois[k] = tuple(int(round(float(t))) for t in v)
            elif isinstance(v, str) and v.count(",") == 3:
                self.rois[k] = tuple(int(round(float(t))) for t in v.split(","))

        self.dirty = False
        self.frame_idx = 0
        self.bgr = None
        self.h = self.w = 0
        self.scale = 1.0

        # 鼠标绘制状态
        self.drawing = False
        self.start = None
        self.cur = None

        self._load_frame(0)

    # ---- 帧 ----
    def _load_frame(self, idx: int):
        idx = max(0, min(idx, self.src.n_frames - 1))
        raw = self.src.read_frame(idx)
        if raw is None:
            return
        self.frame_idx = idx
        if raw.ndim == 2:
            self.bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            self.bgr = raw.copy()
        self.h, self.w = self.bgr.shape[:2]
        sw, sh = screen_size()
        self.scale = fit_scale(self.w, self.h, sw - 120, sh - 220)

    # ---- 鼠标 ----
    def on_mouse(self, event, x, y, flags, param):
        # 显示坐标 -> 原图坐标
        ox, oy = int(x / self.scale), int(y / self.scale)
        ox = max(0, min(ox, self.w - 1))
        oy = max(0, min(oy, self.h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (ox, oy)
            self.cur = (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.cur = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.cur = (ox, oy)          # 以抬起事件自身坐标为准，快速拖拽时更准
            if self.start is not None and self.cur is not None:
                r = normalize_rect(self.start, self.cur)
                if r[2] > r[0] and r[3] > r[1]:
                    self.rois[self.mode] = r
                    self.dirty = True
                    print(f"[标定] {self.mode} = {r}")
                else:
                    print("[标定] 矩形太小，忽略")
            self.start = self.cur = None

    # ---- 绘制 ----
    def render(self) -> np.ndarray:
        vis = self.bgr.copy()

        # 参考线：中心十字
        cv2.line(vis, (self.w // 2, 0), (self.w // 2, self.h - 1), (60, 60, 60), 1)
        cv2.line(vis, (0, self.h // 2), (self.w - 1, self.h // 2), (60, 60, 60), 1)

        for name, rect in self.rois.items():
            if not rect:
                continue
            c = COLORS[name]
            thick = 3 if name == self.mode else 2
            cv2.rectangle(vis, rect[:2], rect[2:], c, thick)
            label = f"{name} {rect[0]},{rect[1]},{rect[2]},{rect[3]}"
            ty = rect[1] + 28 if rect[1] < self.h - 40 else rect[3] - 10
            cv2.putText(vis, label, (rect[0] + 8, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)

        # 正在拖拽的矩形
        if self.drawing and self.start and self.cur:
            cv2.rectangle(vis, self.start, self.cur, (0, 255, 255), 2)

        # HUD
        hud = [
            f"frame {self.frame_idx + 1}/{self.src.n_frames}",
            f"mode: {self.mode}  ({MODE_HINT[self.mode]})",
            "drag=draw  1=social 2=nonsocial  u=undo  r=reset  "
            "a/d=frame  s=save  q=quit",
        ]
        y = 26
        for t in hud:
            cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 3, cv2.LINE_AA)   # 描边
            cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (30, 30, 30), 1, cv2.LINE_AA)
            y += 30

        if self.scale != 1.0:
            vis = cv2.resize(vis, (int(self.w * self.scale), int(self.h * self.scale)),
                             interpolation=cv2.INTER_AREA)
        return vis

    # ---- 保存 ----
    def save(self) -> bool:
        missing = [k for k, v in self.rois.items() if not v]
        if missing:
            print(f"[警告] 尚未标定: {missing}（仍会保存已标定的部分）")

        out = dict(self.cfg)
        for name, rect in self.rois.items():
            out[name] = list(rect) if rect else None
        out["_comment"] = ("由 roi_tool.py 生成；坐标基于原始图像像素 "
                           "x1,y1,x2,y2，左上为原点")
        out["_image_size"] = [self.w, self.h]
        out["_frames_total"] = self.src.n_frames

        try:
            with open(self.out_path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"[错误] 写入失败: {e}")
            return False

        self.dirty = False
        print(f"[保存] ROI 已写入 {self.out_path}")
        return True

    # ---- 主循环 ----
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        # 帧滑块（帧数极多时也能跳）
        maxf = max(0, self.src.n_frames - 1)
        cv2.createTrackbar("frame", WINDOW, self.frame_idx, maxf, lambda v: None)

        print(__doc__)
        print(f"图像: {self.w}x{self.h}, 共 {self.src.n_frames} 帧")
        if self.rois["social"] or self.rois["nonsocial"]:
            print(f"已从配置载入: social={self.rois['social']}, "
                  f"nonsocial={self.rois['nonsocial']}")

        while True:
            # 滑块换帧
            bar = cv2.getTrackbarPos("frame", WINDOW)
            if bar != self.frame_idx:
                self._load_frame(bar)

            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                if self.dirty:
                    print("[提示] 有未保存的修改；再按 q 确认退出，或按 s 保存。")
                    self.dirty = False        # 只提示一次
                    continue
                break
            elif key == ord("s"):
                self.save()
            elif key == ord("u"):
                self.rois[self.mode] = None
                self.dirty = True
            elif key == ord("r"):
                self.rois = {"social": None, "nonsocial": None}
                self.dirty = True
            elif key == ord("a"):
                self._load_frame(self.frame_idx - 1)
                cv2.setTrackbarPos("frame", WINDOW, self.frame_idx)
            elif key == ord("d"):
                self._load_frame(self.frame_idx + 1)
                cv2.setTrackbarPos("frame", WINDOW, self.frame_idx)
            elif 0 <= key < 128 and chr(key) in MODE_KEYS:
                self.mode = MODE_KEYS[chr(key)]
                print(f"[标定] 当前编辑: {self.mode}")

        cv2.destroyAllWindows()
        return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="交互式标定 SPI 分析的 ROI，输出可被 analysis.py --config 复用。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--video", required=True,
                   help="视频文件，或图像序列目录")
    p.add_argument("--pattern", default=None,
                   help="图像序列的文件通配，如 '*.tif'")
    p.add_argument("--fps", type=float, default=None,
                   help="图像序列的帧率（无元数据时必填）")
    p.add_argument("--config", default=None,
                   help="载入已有配置再微调（其它键会原样保留）")
    p.add_argument("--out", default="roi_config.json",
                   help="保存路径（默认 roi_config.json）")
    p.add_argument("--dry-run", action="store_true",
                   help="只载入首帧并打印信息，不开窗口（用于无显示环境自检）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.video):
        print(f"[错误] 路径不存在: {args.video}", file=sys.stderr)
        return 2

    cfg = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"[错误] 配置文件不存在: {args.config}", file=sys.stderr)
            return 2
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

    # 复用 analysis.py 的帧源：自动区分视频 / 图像序列，并处理 16 位 TIFF
    src = analysis.open_source(
        args.video, SimpleNamespace(fps=args.fps, pattern=args.pattern))

    if args.dry_run:
        raw = src.read_frame(0)
        if raw is None:
            print("[错误] 无法读取首帧", file=sys.stderr)
            return 1
        print(f"OK: {src.kind} {src.width}x{src.height}, {src.n_frames} 帧, "
              f"首帧 {raw.shape} {raw.dtype}")
        return 0

    try:
        return RoiPicker(src, args.out, cfg).run()
    finally:
        src.release()


if __name__ == "__main__":
    raise SystemExit(main())
