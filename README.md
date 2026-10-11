# 斑马鱼社交偏好测试 (SPI) 视频分析

从 Pylon Viewer 录制的视频中追踪单条实验鱼，计算社交偏好指数 (SPI)。
纯 CPU、无需训练数据，传统图像处理实现。

```
SPI = (社交区停留时间 − 非社交区停留时间) / (社交区停留时间 + 非社交区停留时间)
```

取值 `[-1, 1]`：`+1` 表示完全偏好社交侧，`-1` 表示完全避开，`0` 表示无偏好。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 最简：自动把画面横向三等分（左=社交区，右=非社交区）
python analysis.py --video test.mp4

# 推荐：显式指定 ROI，并按已知长度标定真实尺度
python analysis.py --video test.mp4 \
    --social 40,60,300,540 --nonsocial 1620,60,1880,540 \
    --calibration-line 40,60,300,60 --calibration-mm 60

# 批量：整个目录共用一份 ROI 配置
python analysis.py --video-dir ./videos --pattern "*.mp4" --config roi.json --out results/
```

不确定 ROI 坐标时，加 `--preview-dir prev/` 会导出背景图和首帧 ROI 叠加图，
用图里量出的像素坐标再回填即可。

## 输出

| 文件 | 内容 |
| --- | --- |
| `out/<视频名>_frames.csv` | 逐帧：帧号、时间、质心 px/cm、所在区、朝向、速度、体长体宽 |
| `out/summary.csv` | 每视频一行：SPI、各区停留时长、进入次数、潜伏期、朝向时间、总路程、追踪率等 |

`summary.csv` 可直接导入 GraphPad Prism / R / Excel 做分组统计。
列名带 `_s` 为秒，`mm_per_px` 为空表示未标定、长度速度以像素为单位。

## 算法要点

1. **背景建模**：等间隔采样若干帧取逐像素**时间中位数**。长期静止的刺激鱼、
   气泡等会被并入背景，从而自然从前景中消失——因此**无需额外目标分类即可
   排除刺激鱼**。
2. **前景提取**：帧差 + 阈值（默认 `mad` 稳健估计），形态学开闭去噪。
3. **连通域**：先膨胀把鱼头鱼尾并成一个连通域，取最大者作实验鱼，避免质心
   在身体两段间跳动。
4. **轨迹后处理**：跳变剔除 → 缺口线性插值 → 滑动平均平滑。
5. **朝向**：由运动方向定鱼头；静止时沿用最近朝向，超时记为缺失。
6. **分区统计**：质心落在哪个 ROI 就累加对应时长，进而得到 SPI 与各项指标。

## 关键参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--social` / `--nonsocial` | 三等分 | ROI，格式 `x1,y1,x2,y2`（左上原点） |
| `--mask` | 无 | 搜索掩膜图，白色区域才允许找鱼，用于手工排除干扰 |
| `--thresh-method` | `mad` | `mad` / `otsu` / `fixed`；低对比度视频优先调 `--thresh` + `fixed` |
| `--mad-k` | 6.0 | 阈值倍数，鱼太小/太暗就调小，噪声多就调大 |
| `--min-area` | 30 | 连通域最小面积，过滤碎点噪声 |
| `--jump-limit` | 0.15 | 相邻帧最大位移（画面对角线比例），抗误检 |
| `--fps` | 视频元数据 | Pylon 导出偶尔缺失帧率，可用此覆盖 |

用 `python analysis.py --save-config roi.json` 生成配置模板填写后，用
`--config` 复用；命令行参数优先级高于配置文件。

`--debug-video dbg/` 导出叠加轨迹、ROI、实时分区的核查视频，用于人工确认
追踪是否正确；`--preview-dir` 导出背景图与 ROI 叠加图。

## 自检

`_selftest/make_synthetic.py` 生成一段合成视频（静止刺激鱼 + 已知轨迹的实验鱼）
和逐帧真值，用于验证算法：

```bash
cd _selftest
python -I make_synthetic.py
python -I ../analysis.py --video synthetic.mp4 \
    --social 0,0,426,540 --nonsocial 853,0,1280,540 --out out
```

合成视频真值 SPI = `-0.0688`，程序输出 `-0.068`，追踪率 100%。

## 已知限制

- 仅适用于**单条实验鱼 + 一条刺激鱼/刺激区**的社交偏好范式。多条鱼自由游动的
  群居 (shoaling) 实验需要多目标追踪，本程序不覆盖。
- 传统方法在鱼与背景对比度极低、或实验鱼长时间静止贴着刺激鱼时可能丢失；
  出现这种情况时用 `--debug-video` 定位，再调阈值或加 `--mask`。
- 若刺激鱼也在缓慢移动，时间中位数吸收不彻底，需用 `--mask` 手工排除。
