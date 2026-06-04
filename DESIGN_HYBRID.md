# HybridCellTracker — 混合架构设计文档

**目标**：结合 EmbedTrack（像素嵌入分割）与 BSGM-CellTrack（查询式追踪）的优势，
构建一个在 CTC 基准上精度更高、时序一致性更强的端到端细胞追踪模型。

---

## 1. 问题分析

| 维度 | EmbedTrack | BSGM-CellTrack | 混合目标 |
|------|-----------|----------------|---------|
| 分割质量 | ✅ 像素级，SEG 高 | ⚠️ 查询式，依赖掩码头 | ✅ 像素嵌入 |
| 追踪一致性 | ⚠️ 近邻匹配，长时漂移 | ✅ track query，长程注意力 | ✅ track query |
| 速度 | ✅ 快（~2s/帧 GPU） | ⚠️ 慢（参数多） | 中等 |
| 不确定性 | ❌ | ✅ 贝叶斯 dropout | ✅ 保留 |
| 细胞分裂 | ⚠️ 仅近邻判断 | ✅ 8D 边界框 | ✅ track query 分裂 |
| 密集场景 | ⚠️ 聚类可能合并 | ✅ 注意力区分 | ✅ 两者互补 |

**核心洞察**：
- 分割用像素嵌入：每个像素预测到所属细胞中心的偏移，通过聚类得到实例掩码
- 追踪用 track query：将上一帧实例的**掩码平均特征**作为 track query 内容，
  通过 Transformer 解码器实现长程时序关联

---

## 2. 整体架构

```
输入帧三元组 [t-1, t, t+1]
        │
        ▼
┌────────────────────────────────────┐
│    Swin-T 主干网络（共享权重）       │  4 阶段 W-MSA/SW-MSA
│    → FPN 特征金字塔               │  [P2, P3, P4, P5]
│    → 多尺度时序 Mamba 融合         │  SSM 跨 T=3 帧
└──────────┬─────────────────────────┘
           │ fused_fpn [P2..P5]
    ┌──────┴──────────────┐
    │                     │
    ▼                     ▼
【分割分支】           【追踪分支】
EmbedSeg               TrackDec
    │                     │
像素解码器          可变形编码器
    │                     │
偏移/带宽/种子度      BSGM 解码器（×N 层）
    │                 ↑         │
    │           track queries    │
    │        ← MaskPoolInit ←    │
    │           (桥接模块)        │
    ▼                            ▼
实例聚类              目标查询输出
（推理时）             分类 + 边界框
    │                     │
    └──────────┬───────────┘
               ▼
        CTC 输出格式
   man_track.txt + mask*.tif
```

---

## 3. 核心桥接模块：MaskPoolQueryInit

这是混合架构的**关键创新点**，连接两个分支。

### 3.1 原理

对于第 t 帧，追踪分支需要 track queries 来关联 t-1 帧的细胞。
传统 BSGM 直接传播上一帧解码器的隐状态。
混合架构改为：用 **t-1 帧的实例掩码** 对 FPN 特征做 **掩码平均池化**，
得到每个实例的区域特征向量作为 track query 内容。

```
mask_avg_pool(fpn_feat_t, mask_m) = Σ_{i∈mask_m} fpn[i] / |mask_m|
```

### 3.2 优势

- 训练时：用 GT 掩码池化，梯度直接流回 FPN（可微）
- 推理时：用预测聚类掩码池化，天然衔接两个分支
- 内容更丰富：比单纯传播查询隐状态包含更多空间纹理信息

### 3.3 实现

```python
class MaskPoolQueryInit(nn.Module):
    """
    Inputs:
        fpn_feats : List[(B, d_model, Hi, Wi)]  FPN 各层特征
        masks     : (B, M, H, W)                上一帧实例掩码（0/1）
        centroids : (B, M, 2)                   实例质心（归一化 cx,cy）
    Returns:
        query_content : (B, M, d_model)          track query 内容
        query_pos     : (B, M, 4)                track query 位置（cx,cy,w,h）
    """
```

---

## 4. 分割分支详解（EmbedSeg）

### 4.1 输出头（每像素）

| 输出 | 维度 | 含义 |
|------|------|------|
| `seg_offsets` | (B, 2, H, W) | dx, dy：像素→细胞中心偏移 |
| `bandwidth`   | (B, 2, H, W) | sx, sy：聚类半径（softplus > 0）|
| `seediness`   | (B, 1, H, W) | 前景置信度 [0,1] |
| `track_offsets`| (B, 2, H, W) | 像素→上一帧细胞中心偏移 |

### 4.2 推理时聚类步骤（来自 EmbedTrack）

```
1. 阈值 seediness → 前景像素集合 F
2. 移位：ê_i = pixel_i + seg_offset_i  （移向细胞中心）
3. 聚类：选取未分配像素，以其 bandwidth 为半径划定簇域
4. 过滤：去除过小的簇
5. 输出：每簇 → 一个实例掩码
```

### 4.3 损失函数

```
L_seg = w_seed · L_seed + w_offset · L_offset + w_bw · L_bw_var
      + w_track · L_track_offset

L_seed   = BCE(seediness, foreground_gt)
L_offset = L1(seg_offsets[fg], gt_offsets[fg])
L_bw_var = Var(bandwidth within each instance)   # 同实例带宽一致性
L_track_offset = Lovász(track_offsets, gt_track_offsets)
```

---

## 5. 追踪分支详解（TrackDec）

### 5.1 查询初始化

**训练时**（用 GT 掩码，可微）：
```
query_content[m] = MaskAvgPool(fpn_t-1, gt_mask_{m,t-1})
query_pos[m]     = centroid_gt_{m,t-1}
```

**推理时**（用预测掩码）：
```
query_content[m] = MaskAvgPool(fpn_t-1, pred_mask_{m,t-1})
query_pos[m]     = centroid_pred_{m,t-1}
```

### 5.2 解码器结构（沿用 BSGMDecoder）

```
for each layer:
    CellGraphLayer (GATv2 kNN)    ← 显式细胞间关系
    QueryMamba (SSM on queries)   ← 时序序列建模
    SelfAttention (masked)        ← track / new 两组分离
    DeformCrossAttention          ← 关注 FPN 多尺度特征
    FFN
```

### 5.3 损失函数

```
L_track = w_cls · L_focal + w_bbox · L_L1 + w_giou · L_GIoU
        + L_div (分裂检测，可选)
```

通过 Hungarian Matcher 与 GT 匹配。

---

## 6. 联合损失

```
L_total = λ_seg · L_seg + λ_track · L_track + λ_aux · Σ L_aux_i
```

默认权重：λ_seg = 1.0, λ_track = 1.0, λ_aux = 0.5

---

## 7. 训练策略（三阶段）

| 阶段 | Epoch | 冻结 | 激活损失 |
|------|-------|------|---------|
| 1. Seg Warmup | 0-7 | 主干 + 追踪分支 | L_seg 仅 |
| 2. Track Warmup | 8-15 | 主干 | L_seg + L_track |
| 3. Full Joint | 16-24 | 无（全参数） | L_total |

---

## 8. 参数量与速度预测

| 模块 | 参数量估算 |
|------|----------|
| Swin-T 主干 + FPN | ~28M |
| 时序 Mamba | ~1M |
| 分割头（EmbedSeg） | ~3M |
| 可变形编码器 | ~2M |
| BSGM 解码器（2层精简）| ~4M |
| **合计** | **~38M** |

速度（GPU A100）：
- BSGM-CellTrack：~180ms/帧
- EmbedTrack：~50ms/帧（GPU）
- **Hybrid 预测**：~220ms/帧（多了分割解码器，但追踪分支更小）

---

## 9. 预期性能提升

| 指标 | EmbedTrack | BSGM | **Hybrid 预期** |
|------|-----------|------|----------------|
| SEG  | 0.90 | 0.88 | **≥ 0.91** |
| TRA  | 0.96 | 0.94 | **≥ 0.97** |
| DET  | —    | 0.92 | **≥ 0.93** |

TRA 提升来源：track query 相比 nearest-neighbor 在长时遮挡和快速运动场景下更鲁棒。
SEG 提升来源：像素嵌入聚类相比查询式掩码头对小细胞边界更精确。

---

## 10. 关键文件

```
src/ai_cstq/models/
├── hybrid_net.py          ← 主模型（本文档对应实现）
│   ├── MaskPoolQueryInit  ← 桥接模块（核心创新）
│   ├── EmbedSegDecoder    ← 分割分支（像素偏移头）
│   └── HybridCellTracker  ← 顶层模型
├── hybrid_criterion.py    ← 联合损失函数
swin_backbone.py           ← 共享主干（复用）
mamba_module.py            ← 时序融合（复用）
bsgm_decoder.py            ← 追踪解码器（复用）
```
