# 斑马鱼社交偏好测试 (SPI) 视频 / 图像序列分析

从 Pylon Viewer 录制或导出的数据中追踪单条实验鱼，计算社交偏好指数 (SPI)。
纯 CPU、无需训练数据，传统图像处理实现。

```
SPI = (社交区停留时间 − 非社交区停留时间) / (社交区停留时间 + 非社交区停留时间)
```

取值 `[-1, 1]`：`+1` 表示完全偏好社交侧，`-1` 表示完全避开，`0` 表示无偏好。

支持两种输入：

- **视频文件**（`.mp4/.avi/.mov/...`）
- **图像序列目录**（Pylon 导出的 `.tif/.png/.bmp/...` 数字编号图），自动识别、
  按自然序排列（`frame_2` 排在 `frame_10` 前），16 位 TIFF 自动按位深映射

## 针对本实验室装置（Basler acA2000-165umNIR, 2048×1088）

真实数据录好后，按以下步骤跑：

```bash
# 1) 先核对 ROI：导出背景图与首帧叠加图
python analysis.py --video data/你的文件.avi --preview-dir prev/

# 2) 打开 prev/*_roi.png，量出社交区/非社交区的像素坐标，
#    填进 roi_template.json（刺激鱼在右侧 → 社交区在右）

# 3) 正式分析
python analysis.py --video data/你的文件.avi --config roi_template.json --out results/
```

**这套数据的关键特点**（已实测确认，roi_template.json 里已按此设好默认值）：

- 实验鱼是**很小的低对比度亮斑**，原始分辨率下仅 15–150 px
  → `--min-area 15`、`--mad-k 5`（不能用通用默认的 30 / 6）
- **刺激鱼静止**，会被时间中位数背景自动吸收，无需手工排除
- 视频 2048×1088 @ ~94.5 fps 约 2 GB；图像序列每张 TIFF 约 2.2 MB
- 用 `--preview-dir` 核对 ROI 是**必做步骤**——鱼缸每次摆放位置都不同，
  `roi_template.json` 里的坐标只是占位示例

> 注意：Pylon 导出的 MJPEG AVI 在顺序读到结尾后 `seek(0)` 会静默失效，
> 程序已在 `rewind` 处做了重开文件处理，无需你关心。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 最简：视频，自动把画面横向三等分（左=社交区，右=非社交区）
python analysis.py --video test.mp4

# 图像序列 + 完整预处理（Pylon 16 位 TIFF 目录）
python analysis.py --video ./seq_dir --fps 30 \
    --crop 0,0,1920,1080 --resize-scale 0.5 \
    --illum-correct --denoise median --clahe

# 显式指定 ROI 并按已知长度标定真实尺度
python analysis.py --video seq_dir --fps 30 \
    --social 40,60,300,540 --nonsocial 1620,60,1880,540 \
    --calibration-line 40,60,300,60 --calibration-mm 60

# 批量：整个父目录（视频文件 + 图像子目录）共用一份配置
python analysis.py --video-dir ./data --config roi.json --out results/
```

> 图像序列没有帧率元数据，**必须用 `--fps` 指定**（不指定会警告并回退 30）。

不知道 ROI 坐标时，加 `--preview-dir prev/` 会导出背景图和首帧 ROI 叠加图，
量出像素坐标再回填即可。

## 预处理（内存中执行，不落中间文件）

固定顺序：**裁剪缩放 → 光照校正 → 降噪 → CLAHE**。

| 参数 | 作用 |
| --- | --- |
| `--crop x1,y1,x2,y2` | 裁掉缸外无关区域，省后续算力 |
| `--resize-scale 0.5` / `--resize 960x540` | 降采样，加速；二选一 |
| `--illum-correct` | 高斯估计光场后做除法拉平，改善四角偏暗/灯反光导致的分割不稳 |
| `--illum-sigma 0.12` | 光照校正核尺度（占短边比例），越大越只保留低频光照 |
| `--denoise median\|gaussian\|bilateral` | 降噪；Pylon 高增益噪点用 `median` |
| `--denoise-ksize 3` | 降噪核尺寸（自动取奇数） |
| `--clahe` | 自适应直方图均衡，拉开鱼与背景灰度差 |

**坐标约定**：`--crop`、`--social`、`--nonsocial`、`--mask`、`--calibration-line`
一律用**原始图像**像素坐标（与你在图像查看器里量到的一致）。启用裁剪/缩放后
程序内部自动换算，输出的质心坐标也换算回原始坐标系。

## 输出

| 文件 | 内容 |
| --- | --- |
| `out/<名称>_frames.csv` | 逐帧：帧号、时间、质心 px/cm、所在区、朝向、速度、体长体宽 |
| `out/summary.csv` | 每输入一行：SPI、各区停留时长、进入次数、潜伏期、朝向时间、总路程、追踪率、预处理描述 |

`summary.csv` 可直接导入 GraphPad Prism / R / Excel 做分组统计。
列名带 `_s` 为秒；`mm_per_px` 为空表示未标定、长度速度以像素为单位。

## 算法要点

1. **背景建模**：等间隔采样若干帧取逐像素**时间中位数**。长期静止的刺激鱼、
   气泡等会被并入背景而自然从前景中消失——因此**无需额外目标分类即可排除
   刺激鱼**。
2. **前景提取**：帧差 + 阈值（默认 `mad` 稳健估计），形态学开闭去噪。
3. **连通域**：先膨胀把鱼头鱼尾并成一个连通域，取最大者作实验鱼，避免质心
   在身体两段间跳动。
4. **轨迹后处理**：跳变剔除 → 缺口线性插值 → 滑动平均平滑。
5. **朝向**：由运动方向定鱼头；静止时沿用最近朝向，超时记为缺失。
6. **分区统计**：质心落在哪个 ROI 就累加对应时长，进而得到 SPI 与各项指标。

## 关键参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--social` / `--nonsocial` | 三等分 | ROI，格式 `x1,y1,x2,y2`（原始坐标） |
| `--mask` | 无 | 搜索掩膜图，白色区域才允许找鱼，用于手工排除干扰 |
| `--thresh-method` | `mad` | `mad` / `otsu` / `fixed`；低对比度视频优先用 `--thresh` + `fixed` |
| `--mad-k` | 6.0 | 阈值倍数，鱼太小/太暗就调小，噪声多就调大 |
| `--min-area` | 30 | 连通域最小面积（原始像素），过滤碎点噪声 |
| `--jump-limit` | 0.15 | 相邻帧最大位移（原始画面对角线比例），抗误检 |
| `--fps` | 视频元数据 / 序列必填 | 序列无元数据，务必指定真实帧率 |

用 `python analysis.py --save-config roi.json` 生成配置模板填写后，用 `--config`
复用（含所有预处理参数）；命令行参数优先级高于配置文件。

`--debug-video dbg/` 导出叠加轨迹、ROI、实时分区的核查视频（原始分辨率）；
`--preview-dir prev/` 导出背景图与 ROI 叠加图。

## 自检

`_selftest/make_synthetic.py` 生成一套等价数据（8 位视频 + 1200 张 16 位 TIFF
序列，静止刺激鱼 + 已知轨迹的实验鱼）和逐帧真值：

```bash
cd _selftest
python -I make_synthetic.py
python -I ../analysis.py --video seq --fps 30 \
    --social 0,0,426,540 --nonsocial 853,0,1280,540 --out out
```

真值 SPI = `-0.0688`：

- 图像序列（无预处理）输出 `-0.0688`，追踪率 100%
- 视频（mp4 有损压缩）输出 `-0.068`
- 裁剪 + 缩放 0.5 + 光照校正下仍输出 `-0.068`，且坐标正确映射回原始系

## 已知限制

- 仅适用于**单条实验鱼 + 一条刺激鱼/刺激区**的社交偏好范式。多条鱼自由游动的
  群居 (shoaling) 实验需要多目标追踪，本程序不覆盖。
- 传统方法在鱼与背景对比度极低、或实验鱼长时间静止贴着刺激鱼时可能丢失；
  出现这种问题用 `--debug-video` 定位，再调阈值、启用预处理或加 `--mask`。
- 若刺激鱼也在缓慢移动，时间中位数吸收不彻底，需用 `--mask` 手工排除。
- Windows 上 OpenCV 无法读写**中文路径**的图像；本项目路径全英文无此问题，
  若你的序列目录名含中文，需先把路径改成英文。
