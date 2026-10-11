# -*- coding: utf-8 -*-
"""
生成合成测试数据 + 真值，用于验证 analysis.py。

产出两套等价数据，真值相同，用于交叉验证：
  1. synthetic.mp4          —— 8 位视频
  2. seq/                   —— 16 位 TIFF 图像序列（模拟 Pylon 导出）

设定：1280x540，30 fps，1200 帧 (40 s)。
  - 刺激鱼：静止长条，贴在左侧刺激区内，全程不动 -> 应被中位数背景吸收。
  - 实验鱼：一个椭圆，按预设时间在社交区/非社交区/中性区停留。
真值由脚本逐帧按质心所在分区统计，绝不读取任何外部文件。
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

W, H, FPS, N = 1280, 540, 30.0, 1200
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SOCIAL = (0, 0, W // 3, H)
NONSOCIAL = (2 * (W // 3), 0, W, H)
STIM = (200, 270)

rng = np.random.default_rng(0)
BG_LEVEL = 40
STIM_LEVEL = 200
FISH_LEVEL = 230
BIT16 = 256          # 16 位值 = 8 位值 * 256


def fish_waypoints():
    """返回 (每帧 x, 每帧 y)。社交区 15 s、非社交区 25 s 的停留设计。"""
    pos = []

    def dwell(target_x, seconds):
        nframes = int(seconds * FPS)
        for i in range(nframes):
            jx = 6.0 * np.sin(2 * np.pi * i / 40.0)
            jy = 6.0 * np.cos(2 * np.pi * i / 37.0)
            pos.append((target_x + jx, 270.0 + jy))

    def travel(x0, x1, seconds):
        nframes = int(seconds * FPS)
        for i in range(nframes):
            f = i / max(nframes - 1, 1)
            pos.append((x0 + (x1 - x0) * f, 270.0 + 20 * np.sin(f * np.pi)))

    travel(640, 150, 2)
    dwell(150, 15)
    travel(150, 1060, 4)
    dwell(1060, 25)
    travel(1060, 640, 2)

    while len(pos) < N:
        pos.append(pos[-1])
    del pos[N:]
    return pos


def render_frame(i, waypoints, sixteen=False):
    frame = np.full((H, W), BG_LEVEL, np.uint8)
    cv2.ellipse(frame, STIM, (34, 12), 0, 0, 360, STIM_LEVEL, -1)
    cx, cy = waypoints[i]
    nx, ny = waypoints[min(i + 2, N - 1)]
    ang = np.degrees(np.arctan2(-(ny - cy), nx - cx if nx != cx else 1e-6))
    cv2.ellipse(frame, (int(cx), int(cy)), (18, 7), ang, 0, 360, FISH_LEVEL, -1)
    noise = rng.normal(0, 2.0, frame.shape)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if sixteen:
        return (frame.astype(np.uint16) * BIT16)
    return frame


def main():
    waypoints = fish_waypoints()

    # --- 1. 视频 ---
    video_path = os.path.join(OUT_DIR, "synthetic.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H), False)
    if not writer.isOpened():
        print("无法创建视频写入器", file=sys.stderr)
        return 1
    for i in range(N):
        writer.write(render_frame(i, waypoints, sixteen=False))
    writer.release()

    # --- 2. 16 位 TIFF 序列 ---
    seq_dir = os.path.join(OUT_DIR, "seq")
    if os.path.isdir(seq_dir):
        shutil.rmtree(seq_dir)
    os.makedirs(seq_dir)
    for i in range(N):
        cv2.imwrite(os.path.join(seq_dir, f"frame_{i:05d}.tif"),
                    render_frame(i, waypoints, sixteen=True))

    # --- 3. 真值（逐帧按质心所在分区统计）---
    t_social = t_nonsocial = 0.0
    for i in range(N):
        x = waypoints[i][0]
        if SOCIAL[0] <= x < SOCIAL[2]:
            t_social += 1 / FPS
        elif NONSOCIAL[0] <= x < NONSOCIAL[2]:
            t_nonsocial += 1 / FPS
    spi_truth = (t_social - t_nonsocial) / (t_social + t_nonsocial)

    truth = {
        "fps": FPS, "n_frames": N,
        "time_social_s": round(t_social, 3),
        "time_nonsocial_s": round(t_nonsocial, 3),
        "spi_truth": round(spi_truth, 4),
        "video": video_path, "seq_dir": seq_dir,
    }
    with open(os.path.join(OUT_DIR, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2, ensure_ascii=False)
    print(json.dumps(truth, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
