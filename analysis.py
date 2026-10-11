# -*- coding: utf-8 -*-
"""
斑马鱼社交偏好测试 (Social Preference Test) 视频 / 图像序列分析
================================================================

从 Pylon Viewer 录制的视频或图像序列中追踪单条实验鱼，计算社交偏好指数 (SPI)。

SPI = (社交区停留时间 - 非社交区停留时间) / (社交区停留时间 + 非社交区停留时间)

取值范围 [-1, 1]：+1 表示完全偏好社交侧，-1 表示完全避开，0 表示无偏好。

处理流程
--------
1. 读取：视频文件，或一个装有序号图像的目录（自动识别、按自然序排列）。
2. 预处理（可选，全部在内存中完成，不写中间文件）：
     裁剪缩放 -> 光照不均校正 -> 降噪 -> CLAHE 对比度增强
3. 时间中位数背景建模。刺激鱼/静止物体若在绝大多数帧中存在，会被并入背景而
   从前景里消失，因此无需额外做目标分类即可自动排除。
4. 前景 = |当前帧 - 背景| >= 阈值，形态学开闭去噪。
5. 连通域分析；用膨胀把断开的鱼头/鱼尾合并，选面积最大的连通域作为实验鱼。
6. 由质心序列得到轨迹；用 minAreaRect 长宽比判定朝向轴，用运动方向定鱼头。
7. 依据质心落在哪个 ROI 累加停留时间，输出逐帧表与汇总表。

坐标约定
--------
所有 ROI、标定长度、`--crop` 都用**原始图像**的像素坐标（左上为原点）给出，
与你在图像查看器里量到的一致。若启用了裁剪/缩放，程序内部自动换算，
输出的质心坐标也换算回原始坐标系。

用法示例
--------
    # 视频，自动按三等分划分社交/中性/非社交区
    python analysis.py --video test.mp4

    # 图像序列（Pylon 导出的 .tif 序号图），并启用完整预处理
    python analysis.py --video ./seq_dir --fps 30 \
        --crop 0,0,1920,1080 --resize-scale 0.5 \
        --illum-correct --denoise median --clahe

    # 显式指定 ROI 与真实尺度
    python analysis.py --video seq_dir --fps 30 \
        --social 40,60,300,540 --nonsocial 1620,60,1880,540 \
        --calibration-line 40,60,300,60 --calibration-mm 60

    # 批量处理某目录下所有输入，共用一份 ROI/预处理配置
    python analysis.py --video-dir ./data --pattern "*.tif" \
        --config roi.json --out results/

配置文件 (JSON) 中可写任意同名参数，命令行参数优先级更高。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

# 视为图像序列的扩展名
IMAGE_EXTS = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg",
              ".pgm", ".ppm", ".pbm")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mjpg")


# --------------------------------------------------------------------------
# 参数定义
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="斑马鱼社交偏好测试 (SPI) 视频/图像序列分析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("输入输出")
    src.add_argument("--video", action="append", default=[],
                     help="输入路径，可重复指定：视频文件，或装有序号图像的目录")
    src.add_argument("--video-dir", default=None,
                     help="输入父目录，处理其中的所有输入；其下的图像子目录按序列处理")
    src.add_argument("--pattern", default=None,
                     help="配合 --video-dir 的文件通配符；缺省自动匹配视频与图像扩展名")
    src.add_argument("--fps", type=float, default=None,
                     help="帧率。图像序列没有帧率元数据，必须用此指定（缺省 30 并警告）；"
                          "视频缺省读取元数据，可用此覆盖")
    src.add_argument("--out", default="results",
                     help="输出目录（不存在则创建）")
    src.add_argument("--config", default=None,
                     help="JSON 配置文件，可定义下方任意参数")
    src.add_argument("--save-config", default=None,
                     help="把本次生效的参数写入指定 JSON 文件后退出")

    prep = p.add_argument_group(
        "预处理（按 裁剪缩放 -> 光照校正 -> 降噪 -> CLAHE 的顺序在内存中执行）")
    prep.add_argument("--crop", default=None,
                      help="裁剪区域 x1,y1,x2,y2（原始图像坐标），先裁掉缸外无关区域")
    prep.add_argument("--resize", default=None,
                      help="缩放后尺寸，格式 WxH，如 960x540（与 --resize-scale 二选一）")
    prep.add_argument("--resize-scale", type=float, default=None,
                      help="缩放倍数，如 0.5 表示降采样一半（与 --resize 二选一）")
    prep.add_argument("--illum-correct", action="store_true",
                      help="光照不均校正：用大核高斯估计光场后做除法拉平，"
                           "改善四角偏暗/灯反光导致的分割不稳")
    prep.add_argument("--illum-sigma", type=float, default=0.12,
                      help="光照校正的核尺度，占图像短边的比例；越大越只保留低频光照")
    prep.add_argument("--denoise", choices=["none", "gaussian", "median", "bilateral"],
                      default="none",
                      help="降噪方式。Pylon 高增益噪点用 median，边缘保真用 bilateral")
    prep.add_argument("--denoise-ksize", type=int, default=3,
                      help="降噪核尺寸（奇数）")
    prep.add_argument("--clahe", action="store_true",
                      help="CLAHE 自适应直方图均衡，拉开鱼与背景的灰度差")
    prep.add_argument("--clahe-clip", type=float, default=2.0,
                      help="CLAHE 对比度限制")
    prep.add_argument("--clahe-grid", type=int, default=8,
                      help="CLAHE 网格数（每边）")

    roi = p.add_argument_group("区域 (ROI)，格式 x1,y1,x2,y2，原始图像坐标")
    roi.add_argument("--social", default=None, help="社交区 (刺激鱼一侧)")
    roi.add_argument("--nonsocial", default=None, help="非社交区 (空白一侧)")
    roi.add_argument("--mask", default=None,
                     help="可选：搜索掩膜图片（原始图像坐标），白色(非0)为允许搜索实验鱼"
                          "的区域，用于手工排除刺激鱼、气泡、水面反光等干扰")

    cal = p.add_argument_group("尺度")
    cal.add_argument("--mm-per-px", type=float, default=None,
                     help="真实尺度（按原始图像像素）。缺省时长度/速度以像素为单位输出")
    cal.add_argument("--calibration-line", default=None,
                     help="已知长度线段 x1,y1,x2,y2（原始图像坐标），配合 --calibration-mm")
    cal.add_argument("--calibration-mm", type=float, default=None,
                     help="上述线段的真实长度 (mm)")

    seg = p.add_argument_group("目标分割")
    seg.add_argument("--thresh", type=float, default=None,
                     help="前景灰度阈值。缺省用 --thresh-method 自动估计")
    seg.add_argument("--thresh-method", choices=["otsu", "fixed", "mad"],
                     default="mad",
                     help="阈值估计方式：otsu 大津法；fixed 固定用 --thresh；"
                          "mad 背景残差的稳健估计(推荐)")
    seg.add_argument("--mad-k", type=float, default=6.0,
                     help="mad 方法的倍数 k，阈值 = 中位数 + k * 1.4826 * MAD")
    seg.add_argument("--min-area", type=float, default=30.0,
                     help="连通域最小面积 (原始图像像素)，过滤碎点噪声")
    seg.add_argument("--max-area", type=float, default=0.0,
                     help="连通域最大面积 (原始图像像素)，0 表示不限制")
    seg.add_argument("--bg-frames", type=int, default=200,
                     help="背景建模采样帧数")
    seg.add_argument("--merge-kernel", type=int, default=7,
                     help="膨胀核尺寸，用于把鱼头鱼尾合并为一个连通域")

    trk = p.add_argument_group("轨迹")
    trk.add_argument("--smooth", type=int, default=5,
                     help="质心平滑窗口 (帧)，用于抑制抖动")
    trk.add_argument("--jump-limit", type=float, default=0.15,
                     help="相邻帧最大位移，单位=原始画面对角线比例；超过判为误检并标缺失")
    trk.add_argument("--max-gap", type=int, default=10,
                     help="缺失帧最大插值长度 (帧)")
    trk.add_argument("--heading-decay", type=int, default=30,
                     help="静止时沿用最近朝向的最大帧数，超出记为 NaN")

    out = p.add_argument_group("输出")
    out.add_argument("--debug-video", default=None,
                     help="可选：导出叠加轨迹/ROI 的核查视频（目录或文件路径）")
    out.add_argument("--preview-dir", default=None,
                     help="可选：导出背景图与首帧 ROI 叠加图的目录")
    out.add_argument("--quiet", action="store_true", help="不打印进度")

    return p


# --------------------------------------------------------------------------
# 参数解析：配置文件 + 命令行合并
# --------------------------------------------------------------------------

_CONFIG_KEYS = (
    "social", "nonsocial", "mask", "mm_per_px", "calibration_line",
    "calibration_mm", "fps", "thresh", "thresh_method", "mad_k",
    "min_area", "max_area", "bg_frames", "merge_kernel", "smooth",
    "jump_limit", "max_gap", "heading_decay",
    # 预处理
    "crop", "resize", "resize_scale", "illum_correct", "illum_sigma",
    "denoise", "denoise_ksize", "clahe", "clahe_clip", "clahe_grid",
)


def load_args(argv=None) -> argparse.Namespace:
    """先扫一遍拿到 --config，把文件里的值作为默认值，再让命令行覆盖。"""
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        unknown = set(raw) - set(_CONFIG_KEYS)
        if unknown:
            print(f"[警告] 配置文件中的未知键被忽略: {sorted(unknown)}",
                  file=sys.stderr)
        parser.set_defaults(**{k: v for k, v in raw.items() if k in _CONFIG_KEYS})

    args = parser.parse_args(argv)

    # 配置文件里用列表写坐标的，统一转成逗号分隔字符串
    for key in ("social", "nonsocial", "calibration_line", "crop"):
        val = getattr(args, key)
        if isinstance(val, (list, tuple)):
            setattr(args, key, ",".join(str(v) for v in val))
    return args


def parse_rect(spec: str, name: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 4:
        raise SystemExit(f"[错误] {name} 需要 4 个数字 x1,y1,x2,y2，得到: {spec!r}")
    try:
        x1, y1, x2, y2 = (int(round(float(p))) for p in parts)
    except ValueError:
        raise SystemExit(f"[错误] {name} 含非数字内容: {spec!r}")
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 == x1 or y2 == y1:
        raise SystemExit(f"[错误] {name} 区域面积为 0: {spec!r}")
    return x1, y1, x2, y2


def parse_line(spec: str, name: str) -> tuple[float, float, float, float]:
    """解析线段坐标，允许水平/垂直退化线段（与 parse_rect 不同）。"""
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 4:
        raise SystemExit(f"[错误] {name} 需要 4 个数字 x1,y1,x2,y2，得到: {spec!r}")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        raise SystemExit(f"[错误] {name} 含非数字内容: {spec!r}")


def parse_size(spec: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", str(spec))
    if not m:
        raise SystemExit(f"[错误] --resize 需要 WxH 格式，如 960x540，得到: {spec!r}")
    return int(m.group(1)), int(m.group(2))


def resolve_mm_per_px(args) -> float | None:
    """把 --mm-per-px / --calibration-line+--calibration-mm 统一成 mm/px。"""
    if args.calibration_line:
        if not args.calibration_mm:
            raise SystemExit("[错误] 指定了 --calibration-line 就必须给 --calibration-mm")
        x1, y1, x2, y2 = parse_line(args.calibration_line, "--calibration-line")
        px = math.hypot(x2 - x1, y2 - y1)
        if px <= 0:
            raise SystemExit("[错误] --calibration-line 两点重合")
        return args.calibration_mm / px
    return args.mm_per_px


# --------------------------------------------------------------------------
# 图像序列读取工具
# --------------------------------------------------------------------------


def natural_key(s: str):
    """自然排序键，保证 frame_2 < frame_10 而非字典序。"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(s))]


def list_sequence(directory: str, pattern: str | None = None,
                  min_count: int = 1) -> list[str]:
    """列出目录中的图像文件，按自然序排列。"""
    if pattern:
        files = glob.glob(os.path.join(directory, pattern))
    else:
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(IMAGE_EXTS)]
    files = sorted([f for f in files if os.path.isfile(f)
                    and f.lower().endswith(IMAGE_EXTS)], key=natural_key)
    return files if len(files) >= min_count else []


def imread_gray(path: str, scale: float = 1.0) -> np.ndarray | None:
    """
    稳健读取单张图像为 8 位灰度。

    Pylon 常导出 16 位 TIFF；这里用 IMREAD_UNCHANGED 读回原始位深，
    再按全局 scale 映射到 8 位，避免逐帧归一化破坏背景差分的一致性。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return img


def estimate_bit_scale(paths: list[str], samples: int = 5) -> float:
    """从若干采样帧估计位深，返回 16->8 位的全局缩放系数。"""
    if not paths:
        return 1.0
    idxs = np.linspace(0, len(paths) - 1, num=min(samples, len(paths))).astype(int)
    maxv = 0
    for i in idxs:
        raw = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if raw.dtype == np.uint8:
            return 1.0
        maxv = max(maxv, int(raw.max()))
    if maxv <= 0:
        return 1.0
    if maxv <= 255:
        return 1.0
    bits = int(math.ceil(math.log2(maxv + 1)))
    return 255.0 / (2 ** bits - 1)


# --------------------------------------------------------------------------
# 帧源：视频 与 图像序列 统一接口
# --------------------------------------------------------------------------


class VideoSource:
    """顺序读取视频文件。"""

    kind = "video"

    def __init__(self, path: str, fps_override: float | None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"[错误] 无法打开视频: {path}")

        self.fps = fps_override or self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0 or math.isnan(self.fps):
            print(f"[警告] {os.path.basename(path)} 帧率无效，回退为 30 fps",
                  file=sys.stderr)
            self.fps = 30.0

        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame

    def read_frame(self, index: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def rewind(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self.cap.release()


class ImageSequenceSource:
    """把一个目录中的图像文件当作视频逐帧读取。"""

    kind = "image-sequence"

    def __init__(self, directory: str, fps: float | None,
                 pattern: str | None = None):
        self.path = directory
        self.files = list_sequence(directory, pattern)
        if not self.files:
            raise SystemExit(f"[错误] 目录中未找到图像文件: {directory}")

        self.n_frames = len(self.files)
        self._scale = estimate_bit_scale(self.files)

        first = imread_gray(self.files[0], self._scale)
        if first is None:
            raise SystemExit(f"[错误] 无法读取首帧: {self.files[0]}")
        self.height, self.width = first.shape[:2]

        if fps is None:
            print(f"[警告] 图像序列无帧率元数据，回退为 30 fps；"
                  f"请用 --fps 指定真实帧率: {directory}", file=sys.stderr)
            fps = 30.0
        self.fps = fps

    def _load(self, i: int):
        return imread_gray(self.files[i], self._scale)

    def frames(self):
        for i in range(self.n_frames):
            f = self._load(i)
            if f is not None:
                yield f

    def read_frame(self, index: int):
        if index < 0 or index >= self.n_frames:
            return None
        return self._load(index)

    def rewind(self):
        pass

    def release(self):
        pass


def open_source(path: str, args) -> VideoSource | ImageSequenceSource:
    """按路径类型自动选择视频或图像序列。"""
    if os.path.isdir(path):
        return ImageSequenceSource(path, args.fps, args.pattern)
    return VideoSource(path, args.fps)


# --------------------------------------------------------------------------
# 预处理（内存中执行，不写中间文件）
# --------------------------------------------------------------------------


class Preprocessor:
    """
    裁剪缩放 -> 光照不均校正 -> 降噪 -> CLAHE。

    ROI / 标定 / --crop 都按原始图像坐标给出，这里负责换算到预处理后的
    工作分辨率，并把检测到的坐标换算回原始坐标系。
    """

    def __init__(self, args, in_w: int, in_h: int):
        # --- 几何：裁剪 ---
        if args.crop:
            cx1, cy1, cx2, cy2 = parse_rect(args.crop, "--crop")
            cx2 = min(cx2, in_w); cy2 = min(cy2, in_h)
            cx1 = min(cx1, cx2 - 1); cy1 = min(cy1, cy2 - 1)
        else:
            cx1, cy1, cx2, cy2 = 0, 0, in_w, in_h
        self.crop_rect = (cx1, cy1, cx2, cy2)
        self.ox, self.oy = cx1, cy1
        cw, ch = cx2 - cx1, cy2 - cy1

        # --- 几何：缩放 ---
        if args.resize and args.resize_scale:
            raise SystemExit("[错误] --resize 与 --resize-scale 只能二选一")
        if args.resize:
            out_w, out_h = parse_size(args.resize)
        elif args.resize_scale:
            s = args.resize_scale
            if s <= 0:
                raise SystemExit("[错误] --resize-scale 必须为正数")
            out_w = max(1, int(round(cw * s)))
            out_h = max(1, int(round(ch * s)))
        else:
            out_w, out_h = cw, ch

        self.out_w, self.out_h = out_w, out_h
        self.sx = out_w / cw
        self.sy = out_h / ch
        self._resize_needed = (out_w, out_h) != (cw, ch)
        self._crop_needed = (cx1, cy1, cx2, cy2) != (0, 0, in_w, in_h)

        # --- 光照校正 ---
        self.illum = bool(args.illum_correct)
        self.illum_sigma = args.illum_sigma

        # --- 降噪 ---
        self.denoise = args.denoise
        k = int(args.denoise_ksize) | 1          # 强制奇数
        self.dk = max(3, k)

        # --- CLAHE ---
        self.clahe = None
        if args.clahe:
            g = max(1, int(args.clahe_grid))
            self.clahe = cv2.createCLAHE(clipLimit=float(args.clahe_clip),
                                         tileGridSize=(g, g))

        self.enabled = any([self._crop_needed, self._resize_needed,
                            self.illum, self.denoise != "none",
                            self.clahe is not None])

    # -- 坐标与尺度换算 --
    def to_original(self, x: float, y: float) -> tuple[float, float]:
        """预处理工作坐标 -> 原始图像坐标。"""
        return x / self.sx + self.ox, y / self.sy + self.oy

    def area_scale(self) -> float:
        """面积换算系数：预处理像素面积 / 原始像素面积。"""
        return self.sx * self.sy

    def length_scale(self) -> float:
        """长度换算系数（近似各向同性）。"""
        return math.sqrt(self.area_scale())

    def apply_geometry_only(self, img: np.ndarray) -> np.ndarray:
        """只做裁剪与缩放，用于掩膜等需要与图像对齐的辅助图。"""
        if self._crop_needed:
            x1, y1, x2, y2 = self.crop_rect
            img = img[y1:y2, x1:x2]
        if self._resize_needed:
            interp = (cv2.INTER_AREA if (self.sx < 1 or self.sy < 1)
                      else cv2.INTER_LINEAR)
            img = cv2.resize(img, (self.out_w, self.out_h),
                             interpolation=interp)
        return img

    # -- 各步骤 --
    def _illum_correct(self, g: np.ndarray) -> np.ndarray:
        h, w = g.shape[:2]
        k = int(max(15, (min(h, w) * self.illum_sigma))) | 1
        k = min(k, (min(h, w) // 2) * 2 - 1)     # 不超过图像尺寸
        k = max(3, k | 1)
        bg = cv2.GaussianBlur(g.astype(np.float32), (k, k), 0)
        bg = np.maximum(bg, 1.0)
        out = g.astype(np.float32) / bg * float(np.mean(g))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _denoise(self, g: np.ndarray) -> np.ndarray:
        if self.denoise == "gaussian":
            return cv2.GaussianBlur(g, (self.dk, self.dk), 0)
        if self.denoise == "median":
            return cv2.medianBlur(g, self.dk)
        if self.denoise == "bilateral":
            return cv2.bilateralFilter(g, self.dk, self.dk * 3, self.dk * 3)
        return g

    def apply(self, g: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return g
        g = self.apply_geometry_only(g)
        if self.illum:
            g = self._illum_correct(g)
        if self.denoise != "none":
            g = self._denoise(g)
        if self.clahe is not None:
            g = self.clahe.apply(g)
        return g

    def describe(self) -> str:
        bits = []
        if self._crop_needed:
            bits.append(f"裁剪({self.ox},{self.oy},{self.out_w},{self.out_h})")
        if self._resize_needed:
            bits.append(f"缩放 x{self.sx:.3f}")
        if self.illum:
            bits.append(f"光照校正(sigma={self.illum_sigma})")
        if self.denoise != "none":
            bits.append(f"降噪({self.denoise} k={self.dk})")
        if self.clahe is not None:
            bits.append("CLAHE")
        return " + ".join(bits) if bits else "无"


class PreprocessedSource:
    """在帧源之上套一层预处理，对外提供与原帧源相同的接口。"""

    def __init__(self, raw, prep: Preprocessor):
        self.raw = raw
        self.prep = prep
        self.path = raw.path
        self.kind = raw.kind
        self.fps = raw.fps
        self.n_frames = raw.n_frames
        self.width = prep.out_w
        self.height = prep.out_h
        self.diag = math.hypot(self.width, self.height)

    def frames(self):
        for f in self.raw.frames():
            yield self.prep.apply(f)

    def read_frame(self, index: int):
        f = self.raw.read_frame(index)
        return None if f is None else self.prep.apply(f)

    def rewind(self):
        self.raw.rewind()

    def release(self):
        self.raw.release()


# --------------------------------------------------------------------------
# 背景建模与阈值
# --------------------------------------------------------------------------


def build_background(src, n_samples: int, quiet: bool
                     ) -> tuple[np.ndarray, float, float]:
    """
    等间隔采样若干帧取逐像素时间中位数，得到背景。

    返回 (背景图, 残差中位数, 残差 MAD)。中位数对刺激鱼这类长期静止的
    目标不敏感，因此它们会自然并入背景而不出现在前景中。
    """
    total = src.n_frames if src.n_frames > 0 else n_samples * 10
    idxs = np.linspace(0, max(total - 1, 0), num=min(max(n_samples, 1), max(total, 1)))
    idxs = sorted({int(round(i)) for i in idxs})

    stack = []
    for i in idxs:
        f = src.read_frame(i)
        if f is not None:
            stack.append(f)
    src.rewind()

    if not stack:
        raise SystemExit(f"[错误] 无法从 {src.path} 读取任何帧")

    if len(stack) == 1:
        bg = stack[0].astype(np.float32)
    else:
        bg = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    resid = np.abs(stack[0].astype(np.float32) - bg)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if not quiet:
        print(f"    背景: 采样 {len(stack)} 帧, 残差中位数 {med:.2f}, MAD {mad:.2f}")
    return bg, med, mad


def compute_threshold(args, med: float, mad: float, bg: np.ndarray) -> float:
    if args.thresh_method == "fixed":
        if args.thresh is None:
            raise SystemExit("[错误] --thresh-method fixed 时必须提供 --thresh")
        return float(args.thresh)

    if args.thresh_method == "otsu":
        img = cv2.normalize(bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(np.clip(t * 0.5, 5.0, 60.0))

    t = med + args.mad_k * 1.4826 * mad
    return float(max(t, 4.0))


# --------------------------------------------------------------------------
# 逐帧目标检测
# --------------------------------------------------------------------------


class Detector:
    """从背景差分中提取实验鱼的质心与外观。工作在预处理后的分辨率。"""

    def __init__(self, args, bg: np.ndarray, thresh: float,
                 mask: np.ndarray | None, area_scale: float = 1.0):
        self.bg = bg
        self.thresh = thresh
        # 把原始像素面积阈值换算到工作分辨率
        self.min_area = max(1.0, args.min_area * area_scale)
        self.max_area = args.max_area * area_scale if args.max_area > 0 else 0.0
        self.mask = mask
        self.k_merge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.merge_kernel, args.merge_kernel))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kc = max(3, args.merge_kernel | 1)
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))

    def foreground(self, gray: np.ndarray) -> np.ndarray:
        if self.bg.dtype == np.uint8:
            diff = cv2.absdiff(gray, self.bg)
        else:
            diff = np.abs(gray.astype(np.float32) - self.bg)
        fg = (diff >= self.thresh).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        if self.mask is not None:
            fg = cv2.bitwise_and(fg, self.mask)
        return fg

    def detect(self, gray: np.ndarray):
        """返回 (cx, cy, area, axis_deg, major, minor) 或 None。"""
        fg = self.foreground(gray)
        merged = cv2.dilate(fg, self.k_merge)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        for c in contours:
            a = cv2.contourArea(c)
            if self.max_area > 0 and a > self.max_area:
                continue
            if best is None or a > best[0]:
                best = (a, c)
        if best is None:
            return None

        merged_area, contour = best
        if merged_area < self.min_area:
            return None

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        major, minor = (w, h) if w >= h else (h, w)
        axis = angle if w >= h else (angle + 90.0)

        raw = cv2.bitwise_and(fg, cv2.drawContours(
            np.zeros_like(fg), [contour], -1, 255, thickness=cv2.FILLED))
        area = float(np.count_nonzero(raw))
        if area < self.min_area:
            area = float(merged_area)

        return float(cx), float(cy), area, float(axis % 180.0), \
            float(major), float(minor)


# --------------------------------------------------------------------------
# 轨迹构建：平滑、跳变剔除、插值、朝向
# --------------------------------------------------------------------------


def smooth_track(xs, ys, window: int):
    if window <= 1:
        return np.asarray(xs, float), np.asarray(ys, float)
    half = window // 2
    n = len(xs)
    sx = np.full(n, np.nan)
    sy = np.full(n, np.nan)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg_x, seg_y = xs[lo:hi], ys[lo:hi]
        ok = ~np.isnan(seg_x)
        if ok.any():
            sx[i] = np.mean(seg_x[ok])
            sy[i] = np.mean(seg_y[ok])
    return sx, sy


def reject_jumps(xs, ys, limit_px: float):
    out_x = np.array(xs, float)
    out_y = np.array(ys, float)
    last = None
    for i in range(len(out_x)):
        if math.isnan(out_x[i]):
            continue
        if last is not None:
            d = math.hypot(out_x[i] - out_x[last], out_y[i] - out_y[last])
            if d / (i - last) > limit_px:
                out_x[i] = out_y[i] = np.nan
                continue
        last = i
    return out_x, out_y


def interpolate_gaps(xs, ys, max_gap: int):
    n = len(xs)
    i = 0
    while i < n:
        if not math.isnan(xs[i]):
            i += 1
            continue
        j = i
        while j < n and math.isnan(xs[j]):
            j += 1
        span = j - i
        if 0 < i and j < n and span <= max_gap:
            for k in range(i, j):
                w = (k - i + 1) / (span + 1)
                xs[k] = xs[i - 1] * (1 - w) + xs[j] * w
                ys[k] = ys[i - 1] * (1 - w) + ys[j] * w
        i = j
    return xs, ys


def compute_headings(xs, ys, moving_speed_px: float, decay: int):
    """估计鱼头朝向角 (度，图像坐标系，0=向右，逆时针为正)。"""
    n = len(xs)
    heading = np.full(n, np.nan)
    last_valid = None
    last_move_idx = None

    for i in range(n):
        if math.isnan(xs[i]):
            continue
        moved = False
        for back in (3, 5, 8):
            j = i - back
            if j >= 0 and not math.isnan(xs[j]):
                dx, dy = xs[i] - xs[j], ys[i] - ys[j]
                if math.hypot(dx, dy) > moving_speed_px * back:
                    heading[i] = math.degrees(math.atan2(-dy, dx)) % 360.0
                    moved = True
                    break
        if not moved and last_valid is not None and last_move_idx is not None:
            if i - last_move_idx <= decay:
                heading[i] = last_valid
        if moved:
            last_valid = heading[i]
            last_move_idx = i
    return heading


# --------------------------------------------------------------------------
# ROI 与统计
# --------------------------------------------------------------------------


def default_rois(w: int, h: int):
    """缺省横向三等分：左=社交区，中=中性区，右=非社交区。"""
    third = w // 3
    return (0, 0, third, h), (2 * third, 0, w, h)


def zone_of(x: float, y: float, social, nonsocial) -> str:
    if social and social[0] <= x < social[2] and social[1] <= y < social[3]:
        return "social"
    if nonsocial and nonsocial[0] <= x < nonsocial[2] and nonsocial[1] <= y < nonsocial[3]:
        return "nonsocial"
    return "neutral"


def count_entries(zones, target: str) -> int:
    entries = 0
    prev = False
    for z in zones:
        cur = (z == target)
        if cur and not prev:
            entries += 1
        prev = cur
    return entries


def latency(zones, dt: float, target: str) -> float:
    for i, z in enumerate(zones):
        if z == target:
            return (i + 1) * dt
    return float("nan")


def _r(v: float):
    return round(v, 3) if not math.isnan(v) else "NaN"


# --------------------------------------------------------------------------
# 单输入主流程
# --------------------------------------------------------------------------


def analyze_video(path: str, args, out_dir: str) -> dict:
    name = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]
    if not args.quiet:
        print(f"\n>>> 处理 {os.path.basename(os.path.normpath(path))}")

    raw = open_source(path, args)
    prep = Preprocessor(args, raw.width, raw.height)
    src = PreprocessedSource(raw, prep)
    dt = 1.0 / src.fps

    if not args.quiet:
        print(f"    {raw.kind}: {raw.width}x{raw.height} -> "
              f"{src.width}x{src.height}, {src.fps:.2f} fps, {raw.n_frames} 帧")
        if prep.enabled:
            print(f"    预处理: {prep.describe()}")

    # --- 区域（原始图像坐标）---
    social = parse_rect(args.social, "--social") if args.social else None
    nonsocial = parse_rect(args.nonsocial, "--nonsocial") if args.nonsocial else None
    if social is None or nonsocial is None:
        d_soc, d_non = default_rois(raw.width, raw.height)
        if social is None:
            social = d_soc
            print(f"[警告] 未指定 --social，缺省使用左侧 1/3: {social}，"
                  f"请确认是否符合你的装置", file=sys.stderr)
        if nonsocial is None:
            nonsocial = d_non
            print(f"[警告] 未指定 --nonsocial，缺省使用右侧 1/3: {nonsocial}，"
                  f"请确认是否符合你的装置", file=sys.stderr)

    # 掩膜：用与图像相同的几何变换，使其落在工作分辨率上
    search_mask = None
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise SystemExit(f"[错误] 无法读取掩膜: {args.mask}")
        if (m.shape[1], m.shape[0]) != (raw.width, raw.height):
            m = cv2.resize(m, (raw.width, raw.height),
                           interpolation=cv2.INTER_NEAREST)
        m = prep.apply_geometry_only(m)
        search_mask = (m > 0).astype(np.uint8) * 255

    # --- 背景与阈值（在工作分辨率上）---
    bg, med, mad = build_background(src, args.bg_frames, args.quiet)
    thresh = compute_threshold(args, med, mad, bg)
    if not args.quiet:
        print(f"    前景阈值 = {thresh:.2f} ({args.thresh_method})")

    detector = Detector(args, bg, thresh, search_mask, prep.area_scale())
    mm_per_px = resolve_mm_per_px(args)          # 按原始像素
    lscale = prep.length_scale()
    ascale = prep.area_scale()

    # --- 逐帧检测（结果换算回原始坐标）---
    xs, ys, areas, axes, majors, minors, vals = [], [], [], [], [], [], []
    for gray in src.frames():
        r = detector.detect(gray)
        if r is None:
            xs.append(np.nan); ys.append(np.nan); areas.append(np.nan)
            axes.append(np.nan); majors.append(np.nan); minors.append(np.nan)
            vals.append(0)
        else:
            cx, cy, a, ax, mj, mn = r
            ox_, oy_ = prep.to_original(cx, cy)
            xs.append(ox_); ys.append(oy_)
            areas.append(a / ascale)
            axes.append(ax)
            majors.append(mj / lscale)
            minors.append(mn / lscale)
            vals.append(1)

    n = len(xs)
    if n == 0:
        raise SystemExit(f"[错误] {path} 未读到任何帧")
    xs = np.array(xs); ys = np.array(ys); areas = np.array(areas)
    axes = np.array(axes); majors = np.array(majors); minors = np.array(minors)
    vals = np.array(vals)
    raw_detected = int(np.nansum(vals))

    # --- 轨迹清理（原始坐标）---
    raw_diag = math.hypot(raw.width, raw.height)
    xs, ys = reject_jumps(xs, ys, args.jump_limit * raw_diag)
    xs, ys = interpolate_gaps(xs, ys, args.max_gap)
    xs, ys = smooth_track(xs, ys, args.smooth)
    valid = ~np.isnan(xs)

    # --- 朝向与速度 ---
    moving_px = max(2.0, 0.004 * raw_diag)
    headings = compute_headings(xs, ys, moving_px, args.heading_decay)

    speeds = np.full(n, np.nan)
    for i in range(1, n):
        if valid[i] and valid[i - 1]:
            speeds[i] = math.hypot(xs[i] - xs[i - 1],
                                   ys[i] - ys[i - 1]) / dt

    # --- 分区与指标 ---
    zones = [zone_of(xs[i], ys[i], social, nonsocial) if valid[i] else "absent"
             for i in range(n)]

    t_social = sum(dt for z in zones if z == "social")
    t_nonsocial = sum(dt for z in zones if z == "nonsocial")
    t_neutral = sum(dt for z in zones if z == "neutral")
    t_absent = sum(dt for z in zones if z == "absent")

    denom = t_social + t_nonsocial
    spi = (t_social - t_nonsocial) / denom if denom > 0 else float("nan")
    valid_time = t_social + t_nonsocial + t_neutral
    spi_total = ((t_social - t_nonsocial) / valid_time
                 if valid_time > 0 else float("nan"))

    # 在社交区内朝向社交侧的时间
    stim_x = (social[0] + social[2]) / 2.0
    stim_y = (social[1] + social[3]) / 2.0
    orient_time = 0.0
    for i in range(n):
        if zones[i] != "social" or math.isnan(headings[i]):
            continue
        ang = math.degrees(math.atan2(-(stim_y - ys[i]), stim_x - xs[i])) % 360.0
        if abs((headings[i] - ang + 180.0) % 360.0 - 180.0) <= 45.0:
            orient_time += dt

    d_soc = [math.hypot(xs[i] - stim_x, ys[i] - stim_y)
             for i in range(n) if valid[i]]
    mean_dist_px = float(np.mean(d_soc)) if d_soc else float("nan")
    mean_dist = mean_dist_px * mm_per_px if mm_per_px else mean_dist_px

    unit = "mm" if mm_per_px else "px"
    scale = mm_per_px if mm_per_px else 1.0
    total_dist = float(np.nansum([s * dt for s in speeds if not math.isnan(s)]))
    mean_speed = (float(np.nanmean(speeds)) if np.any(~np.isnan(speeds))
                  else float("nan"))

    summary = {
        "input": os.path.basename(os.path.normpath(path)),
        "kind": raw.kind,
        "preprocess": prep.describe(),
        "fps": round(src.fps, 4),
        "n_frames": n,
        "duration_s": round(n * dt, 3),
        "detected_frames": raw_detected,
        "valid_frames": int(valid.sum()),
        "tracking_rate": round(valid.sum() / n, 4),
        "time_social_s": round(t_social, 3),
        "time_nonsocial_s": round(t_nonsocial, 3),
        "time_neutral_s": round(t_neutral, 3),
        "time_absent_s": round(t_absent, 3),
        "spi": round(spi, 4) if not math.isnan(spi) else "NaN",
        "spi_total": round(spi_total, 4) if not math.isnan(spi_total) else "NaN",
        "entries_social": count_entries(zones, "social"),
        "entries_nonsocial": count_entries(zones, "nonsocial"),
        "latency_social_s": _r(latency(zones, dt, "social")),
        "latency_nonsocial_s": _r(latency(zones, dt, "nonsocial")),
        "orient_time_s": round(orient_time, 3),
        "orient_ratio_in_social": (round(orient_time / t_social, 4)
                                   if t_social > 0 else "NaN"),
        "total_distance": round(total_dist * scale, 3),
        "mean_speed": round(mean_speed * scale, 3),
        "mean_dist_to_social_center": round(mean_dist, 3),
        "unit": unit,
        "mm_per_px": round(mm_per_px, 6) if mm_per_px else "",
        "social_roi": ",".join(map(str, social)),
        "nonsocial_roi": ",".join(map(str, nonsocial)),
        "thresh_used": round(thresh, 3),
    }

    # --- 逐帧表 ---
    os.makedirs(out_dir, exist_ok=True)
    frames_path = os.path.join(out_dir, f"{name}_frames.csv")
    with open(frames_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "x_px", "y_px", "x_cm", "y_cm",
                    "zone", "valid", "area_px", "axis_deg", "heading_deg",
                    "speed_mm_s", "body_length_mm", "body_width_mm"])
        for i in range(n):
            w.writerow([
                i, round(i * dt, 4),
                round(xs[i], 2) if valid[i] else "",
                round(ys[i], 2) if valid[i] else "",
                round(xs[i] * mm_per_px / 10.0, 4) if valid[i] and mm_per_px else "",
                round(ys[i] * mm_per_px / 10.0, 4) if valid[i] and mm_per_px else "",
                zones[i], int(valid[i]),
                round(areas[i], 1) if not math.isnan(areas[i]) else "",
                round(axes[i], 1) if not math.isnan(axes[i]) else "",
                round(headings[i], 1) if not math.isnan(headings[i]) else "",
                round(speeds[i] * mm_per_px, 3)
                if mm_per_px and not math.isnan(speeds[i]) else "",
                round(majors[i] * mm_per_px, 3)
                if mm_per_px and not math.isnan(majors[i]) else "",
                round(minors[i] * mm_per_px, 3)
                if mm_per_px and not math.isnan(minors[i]) else "",
            ])

    # --- 可选输出（渲染在原始分辨率上）---
    if args.debug_video:
        _write_debug_video(raw, out_dir, name, xs, ys, zones, headings,
                           social, nonsocial, args)
    if args.preview_dir:
        _write_previews(raw, args.preview_dir, name, social, nonsocial, args)

    src.release()
    if not args.quiet:
        print(f"    SPI = {summary['spi']}  "
              f"(社交 {t_social:.1f}s / 非社交 {t_nonsocial:.1f}s, "
              f"追踪率 {summary['tracking_rate']:.1%})")
        print(f"    写出 {frames_path}")
    return summary


def _write_debug_video(raw, out_dir, name, xs, ys, zones, headings,
                       social, nonsocial, args):
    path = args.debug_video
    if (os.path.isdir(path) or path.endswith(("/", "\\"))
            or not path.lower().endswith(VIDEO_EXTS)):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"{name}_debug.mp4")

    raw.rewind()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             raw.fps, (raw.width, raw.height))
    if not writer.isOpened():
        print(f"[警告] 无法创建核查视频（编码器不可用？）: {path}", file=sys.stderr)
        raw.rewind()
        return

    colors = {"social": (0, 180, 0), "nonsocial": (0, 0, 220),
              "neutral": (150, 150, 150), "absent": (60, 60, 60)}
    i = 0
    for gray in raw.frames():
        if gray.ndim == 2:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            vis = gray.copy()
        cv2.rectangle(vis, social[:2], social[2:], (0, 200, 0), 2)
        cv2.rectangle(vis, nonsocial[:2], nonsocial[2:], (0, 0, 200), 2)
        trail = [(int(xs[j]), int(ys[j])) for j in range(max(0, i - 60), i + 1)
                 if not math.isnan(xs[j])]
        for a, b in zip(trail, trail[1:]):
            cv2.line(vis, a, b, colors.get(zones[i], (150, 150, 150)), 2)
        if not math.isnan(xs[i]):
            p = (int(xs[i]), int(ys[i]))
            cv2.circle(vis, p, 4, (0, 255, 255), -1)
            if not math.isnan(headings[i]):
                a = math.radians(headings[i])
                tip = (int(p[0] + 30 * math.cos(a)), int(p[1] - 30 * math.sin(a)))
                cv2.arrowedLine(vis, p, tip, (0, 255, 255), 2, tipLength=0.4)
        cv2.putText(vis, f"f={i} zone={zones[i]}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(vis)
        i += 1
    writer.release()
    raw.rewind()


def _write_previews(raw, preview_dir, name, social, nonsocial, args):
    """导出背景图（原始分辨率）与首帧 ROI 叠加图，便于核对坐标。"""
    os.makedirs(preview_dir, exist_ok=True)

    # 背景在原始分辨率上重算（工作分辨率的背景不便于对照坐标）
    bg_raw, _, _ = build_background(raw, min(args.bg_frames, 50), True)
    bgu = cv2.normalize(bg_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(preview_dir, f"{name}_background.png"), bgu)

    first = raw.read_frame(0)
    raw.rewind()
    if first is not None:
        if first.ndim == 2:
            vis = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
        else:
            vis = first.copy()
        cv2.rectangle(vis, social[:2], social[2:], (0, 200, 0), 2)
        cv2.rectangle(vis, nonsocial[:2], nonsocial[2:], (0, 0, 200), 2)
        cv2.putText(vis, "social", (social[0] + 8, social[1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(vis, "nonsocial", (nonsocial[0] + 8, nonsocial[1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        cv2.imwrite(os.path.join(preview_dir, f"{name}_roi.png"), vis)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def collect_inputs(args) -> list[str]:
    items = list(args.video)

    if args.video_dir:
        d = args.video_dir
        if args.pattern:
            items.extend(sorted(glob.glob(os.path.join(d, args.pattern))))
        else:
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                if os.path.isfile(fp) and (f.lower().endswith(VIDEO_EXTS)
                                           or f.lower().endswith(IMAGE_EXTS)):
                    items.append(fp)
        # 子目录若含图像序列，也作为输入
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            # 至少 10 张图才当作序列，避免把预览图/输出目录误当输入
            if os.path.isdir(fp) and list_sequence(fp, min_count=10):
                items.append(fp)

    seen, out = set(), []
    for v in items:
        rp = os.path.abspath(v)
        if rp not in seen:
            seen.add(rp)
            out.append(v)
    return out


def main(argv=None) -> int:
    args = load_args(argv)

    if args.save_config:
        cfg = {k: getattr(args, k) for k in _CONFIG_KEYS
               if getattr(args, k) is not None}
        with open(args.save_config, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)
        print(f"已写出配置模板: {args.save_config}")
        print("请填入 --social / --nonsocial 等区域坐标后，用 --config 复用。")
        return 0

    inputs = collect_inputs(args)
    if not inputs:
        print("[错误] 未指定输入。用 --video 或 --video-dir 指定视频/图像序列目录。",
              file=sys.stderr)
        return 2

    summaries = []
    for v in inputs:
        if not os.path.exists(v):
            print(f"[警告] 路径不存在，跳过: {v}", file=sys.stderr)
            continue
        try:
            summaries.append(analyze_video(v, args, args.out))
        except SystemExit:
            raise
        except Exception as exc:                # 批量处理时不因单个输入中断
            print(f"[错误] {v} 处理失败: {exc}", file=sys.stderr)

    if not summaries:
        print("[错误] 没有成功处理任何输入", file=sys.stderr)
        return 1

    summary_path = os.path.join(args.out, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    print(f"\n完成：{len(summaries)} 个输入，汇总表 -> {summary_path}")
    if len(summaries) > 1:
        spis = [s["spi"] for s in summaries if isinstance(s["spi"], (int, float))]
        if spis:
            sd = np.std(spis, ddof=1) if len(spis) > 1 else 0.0
            print(f"SPI: 均值 {np.mean(spis):.4f}, 标准差 {sd:.4f}, "
                  f"范围 [{min(spis):.4f}, {max(spis):.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
