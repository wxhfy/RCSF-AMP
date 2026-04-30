# SUCF-AMP D1 论文实验补强 TODO (v2)

> 本文档基于 `reviews.md` 中审稿人 4 大核心质疑，整合 `todo.md` 中的散点议题，
> 全面规划 D1 论文的实验补强工作。下一阶段的所有实验都应以此为单一来源。

---

## 0. 整体目标与论证链

把论文证据链补成一个完整闭环：

> **问题真实存在 → 低置信结构污染可直接诊断 → SUCF-AMP 能缓解污染 → 优势在数据集 / 结构质量 / 预测器 / 任务上都成立**

只要这条链完整,后面无论投 PR 还是其他期刊都比现在更稳。

---

## 1. 审稿人 4 大核心质疑（必须正面回应）

来自 `reviews.md` 的精炼:

| 编号 | 质疑核心 | 回应方向 |
|------|---------|---------|
| **R3-1** | "feature contamination" 证据偏间接 | 直接诊断实验:扰动 low-pLDDT 残基,观察预测/表示敏感性 |
| **R3-2** | static fusion 不足的论证偏概念化 | 补 uncertainty-aware baseline 群,证明不是"塞 pLDDT 就行" |
| **R4-6** | 比较实验在 B1、消融在 B2,设置不一致 | 双 benchmark 上 comparison + ablation + robustness 全做 |
| **R4-4&5** | 方法图缺关键模块、方法叙述像公式流水账 | 重画 Figure 2 拆分 SCGC/SGFN 子图、重写方法章节 |

R3 的 "first work" 收窄、R4 的 "可推广到其他短肽" 证据补强,属于次级问题(随实验补完后改文字)。

---

## 2. 优先级与执行顺序

- **P0 必做**:直接回应审稿人核心质疑,论文上线前必须完成
- **P1 强烈建议**:进一步强化论证链与机制可解释性
- **P2 有余力做**:完善泛化叙事,作为亮点而非主线

整体执行顺序:**先做 P0,再 P1,P2 视时间补**。所有实验做完后再统一改文案、重画图。

---

# P0:必做实验(直接回应审稿)

## P0-1: 低 pLDDT 区域定向扰动 / 直接污染诊断(回应 R3-1)

### 目的
直接证明:低置信度结构区域的几何信息会污染表示与预测,而 SUCF-AMP 能缓解这种污染。

### 实验设计
对 B1、B2 的测试集,每个样本构造 4 类扰动:

1. **Low-pLDDT 定向扰动**:仅扰动 pLDDT < 70 的残基
2. **High-pLDDT 定向扰动**(对照):仅扰动 pLDDT ≥ 70 的残基,数量与 1 相同
3. **随机扰动**(对照):随机选择相同数量残基扰动
4. **零扰动**(基线):原始输入

每类扰动至少做两种方式:
- mask / feature zero-out(节点特征置零)
- 噪声注入(节点特征 Gaussian 噪声 / 边删除 / 坐标 jitter)

### 对比模型
- SUCF-AMP (Full)
- - SCGC
- - pLDDT Gate
- - SGFN(替换为 Concat)
- - Structure(纯序列)
- 至少一个最强 multimodal baseline(SSFGM-Model)

### 输出指标
- ΔACC / ΔAUC / ΔMCC(扰动前后差)
- logit change(均值 ± 标准差)
- probability change
- label flip rate
- representation drift:cosine distance / CKA / centroid drift

### 预期结论
- 普通 multimodal 模型在 low-pLDDT 定向扰动下退化最严重
- SUCF-AMP 在 low-pLDDT 扰动下退化最小
- 两者在 high-pLDDT 扰动下退化相当
→ 证明低置信结构确实造成污染,且 SCGC 真在抑制

### 产出
- `experiments/phase1_contamination_diagnosis/`(已有目录,扩展)
- 表:label flip rate + 显著性
- 柱状图:不同扰动下性能下降幅度
- 箱线图:logit change / representation drift

---

## P0-2: 更强 uncertainty-aware baseline 补齐(回应 R3-2)

### 目的
证明 SUCF-AMP 的 SCGC + SGFN 设计 **不是** "把 pLDDT 喂进去就行",
而是确实更有效。

### 要补的 baseline 群
| Baseline | 设计 |
|---------|------|
| **B-naive-1** | pLDDT 直接拼到节点特征(类似 sAMPpred-GAT 但显式) |
| **B-naive-2** | pLDDT 加权 mean pooling |
| **B-naive-3** | pLDDT 作为 attention bias 加在 GAT |
| **B-naive-4** | Simple confidence-gated fusion(去掉 SCGC/SGFN,只在 concat 前做 pLDDT 门控) |
| **B-naive-5** | 去 sequence guidance,保留 pLDDT gate |
| **B-naive-6** | 去 pLDDT gate,保留 sequence-guided structure attention |
| **B-naive-7** | uncertainty-aware concat(pLDDT 加权拼接) |
| **B-naive-8** | uncertainty-aware attention(把 pLDDT 直接接入 cross-attn 的 KQV) |

### 输出指标
- B1 / B2 上的 ACC / AUC / MCC(5 seeds)
- 配对 t-test vs SUCF-AMP

### 预期结论
- B-naive-* 在 B1 / B2 上仍然落后于 SUCF-AMP
- 在低质量 pLDDT 子集上差距更大
→ 证明 calibrate-then-fuse 比 simple uncertainty injection 更有效

### 产出
- 主结果对照表(双 benchmark)
- 显著性标注

---

## P0-3: 双 benchmark 实验一致性(回应 R4-6)

### 目的
解决"comparison 在 B1、ablation 在 B2"的不一致问题。

### 子任务
**P0-3-A: B1 上的完整消融**
- - Structure
- - SCGC(整体)
- - Seq Guide(SCGC 内只去 sequence guidance)
- - pLDDT Gate
- - SGFN(替换为 Concat)
- - GRU Gate(SGFN 内只去 GRU)
- - Bi-Mamba(替换为 Transformer)

**P0-3-B: B2 上的完整 SOTA 比较**
- 与 Table 1 同款 baseline 集合
- 包括 SSFGM-Model、PepNet、AMP-BERT、sAMPpred-GAT 等

**P0-3-C: 双 benchmark 的结构质量分层 robustness**
- 二分:mean pLDDT > 70 vs ≤ 70
- 四分:0–50 / 50–70 / 70–90 / 90–100
- 对比模型:SUCF-AMP / strongest multimodal baseline / -Structure

### 输出指标
- ACC / AUC / SEN / SPEC / MCC(5 seeds, mean ± std)
- 高低质量之间的 drop
- 配对 t-test 显著性

### 产出
- 主结果表 ×2(B1, B2)
- 消融表 ×2(B1, B2)
- robustness 表 + 分层曲线 ×2

---

## P0-4: 方法图 + 方法叙述重写(回应 R4-4 / R4-5)

### 目的
让审稿人直观看懂 SCGC、SGFN、Bi-Mamba 在做什么,而不是只看公式。

### 子任务

**P0-4-A: 重画 Figure 2(总框架)**
- 明确标注所有中间表示:H_struct, H'_struct, H_raw, H_seq, H'_seq, H''_struct, H_fused, H_ctx
- 用箭头说明每个中间表示如何流动
- 画出 GRU、Bi-Mamba、Cross-Attention 的内部连接

**P0-4-B: 新增 Figure 2.A(SCGC 详细图)**
- pLDDT gate 的具体计算过程
- sequence-guided R-GAT 的 K 层迭代
- 高/低 pLDDT 区域的不同处理可视化

**P0-4-C: 新增 Figure 2.B(SGFN 双向融合图)**
- forward pass:H'_struct as Q → CrossAttn(H_seq) → GRU → H'_seq
- backward pass:H'_seq as Q → CrossAttn(H_struct) → H''_struct
- 三路 concat 形成 H_fused

**P0-4-D: 重写方法章节叙述**
按以下模板组织 SCGC / SGFN / Bi-Mamba 各小节:
1. 一段自然语言说明"为什么需要这个模块"(核心动机)
2. 一段说明"输入是什么、输出是什么、解决什么问题"
3. 公式
4. 公式后一句直观解释
5. 与上下游模块的边界

参考审稿人示例:
> "SCGC is designed to prevent unreliable structural regions from
> dominating multimodal fusion. It first uses sequence-guided graph
> attention to refine local structural features, and then applies a
> pLDDT-conditioned gate to interpolate between structural and
> sequence representations residue by residue."

### 产出
- `supplementary_figures/figure2_main.{pdf,svg}`
- `supplementary_figures/figure2A_scgc_detail.{pdf,svg}`
- `supplementary_figures/figure2B_sgfn_detail.{pdf,svg}`
- 重写的 `IF_main.tex` 方法章节

---

# P1:强烈建议(强化论证链)

## P1-1: calibrate-then-fuse 顺序合理性实验

### 目的
证明 SCGC → SGFN 的顺序不是拍脑袋。

### 对照
- Full:calibrate-then-fuse(默认)
- fuse-then-calibrate
- no calibration, direct guided fusion
- no guided fusion, only concat

### 指标
- 主性能(ACC / AUC / MCC)
- robustness 指标
- P0-1 的污染敏感性指标

### 产出
- 模块顺序对照表(B1, B2)

参考代码:`experiments/calibration_order_experiment.py`(已有,需扩充到双 benchmark)

---

## P1-2: pLDDT gate 行为分析

### 目的
证明 gate 真在按结构置信度调节,不是黑盒权重。

### 分析
- gate value vs pLDDT 的 Pearson / Spearman 相关性
- 高 / 中 / 低 pLDDT 区间的 gate 分布
- 正确预测 vs 错误预测样本的 gate pattern
- 正样本 vs 负样本的 gate pattern

### 产出
- gate-pLDDT 散点图 + 相关系数
- 分桶柱状图
- 失败案例 gate 分布对比

参考代码:`experiments/analyze_plddt_gate_behavior.py`(已有)

---

## P1-3: SGFN 融合行为分析

### 目的
证明 guided fusion 在不同结构质量下确实发生模态依赖切换。

### 分析方法(任选 1–2 种)
- attention weight 分析:high-quality vs low-quality 样本
- cross-modal contribution score
- integrated gradients
- ablation-based modality reliance score

### 产出
- attention heatmap
- modality reliance 对比图(高 vs 低 pLDDT)

---

## P1-4: 多不确定性信号验证(整合 todo.md)

### 目的
证明框架不依赖单一不确定性度量(pLDDT)。

### 实验
- 把 pLDDT 替换为 pTM
- 把 pLDDT 替换为 ipTM
- 把 pLDDT 与 pTM 组合(early stopping 选优)

### 输出
- 不确定性信号互换表(MCC / AUPRC / robustness drop)

---

## P1-5: 5-seed 统计显著性补齐(整合 todo.md)

### 目的
回应"是否只是偶然提升"的质疑。

### 要做
- 统一 seeds:{37, 42, 123, 456, 789}
- Full vs best baseline:配对 t-test 或 Wilcoxon
- Full vs key ablation:同上
- robustness drop:bootstrap 置信区间(95% CI)

### 输出
- p-value 表
- CI 标注
- 主结果 / 消融 / robustness 全部加显著性标记

---

## P1-6: 不平衡指标补充(整合 todo.md)

### 目的
B2 是不平衡数据集,需要更合适的指标。

### 要做
- B2 主结果 + 消融:ACC, AUC, **AUPRC**, MCC, SEN, SPEC, F1
- 画 PR 曲线
- 主指标改为 MCC + AUPRC 并列

---

## P1-7: 强 baseline 补位(整合 todo.md)

### 要补的 baseline
- **sAMPpred-GAT 复现**:统一结构预测管线(ESMFold)下重训
- **双模对比学习**:SimCLR / InfoNCE 风格的序列-结构对齐(对标 Kong 2024)

### 输出
- 主结果表附录补位

---

## P1-8: 分层与误差分析(整合 todo.md)

### 分层维度
- 按 low-pLDDT 残基比例分桶
- 按肽长度分桶
- 按净电荷 / 疏水性分桶
- 按家族簇分桶

### 误差分析
- 混淆矩阵
- 2 例典型失败案例剖析(原因分析 → 改进方向)
- 残基 pLDDT 热图可视化

### 输出
- 分层性能表
- 失败案例 figure

---

## P1-9: 数据分析(整合 todo.md)

### 要做
- ESMFold 与 AlphaFold2 在 AMP / non-AMP 上的 pLDDT 分布对比
- 所有 AMP 中局部 pLDDT < 70 占比统计
- 肽长度 vs 平均 pLDDT 散点图
- B1 / B2 的 pLDDT 分布、长度分布、电荷分布对比

### 输出
- 数据分析章节(论文 Section 4.1.1 末尾或 Appendix)
- 分布图集合

---

# P2:有余力做(泛化叙事)

## P2-1: 其他短肽任务迁移(整合 todo.md)

### 候选任务
- 抗病毒肽预测
- 抗癌肽预测
- 毒性肽(已有 ToxGIN 对照,补完整 protocol)
- 溶血性肽预测

### 要做(每个任务)
- 数据集来源 + 划分 protocol(明确)
- 至少 3 个 baseline
- 主指标 + 结构质量分层指标

### 输出
- 主表(论文主文)而非 supplementary
- 至少 1 个任务的 robustness 分析

---

## P2-2: AMP 多标签子类型预测

预测同时:抗菌 + 抗真菌 + 抗病毒 + 抗肿瘤多标签任务。
通过换头(任务头)实现,主干保持 SUCF-AMP。

---

## P2-3: OmegaFold 第三个预测器(整合 todo.md)

把 cross-predictor 分析扩展到 ESMFold + AlphaFold2 + OmegaFold。

---

## P2-4: Reliability calibration 指标

证明不仅分类性能更好,预测置信度也更合理。

### 指标
- ECE (Expected Calibration Error)
- Brier Score
- Reliability diagram

---

# 论文实验章节最终结构(实验做完后写)

```
4. Experiments
  4.1 Experimental Setup
  4.2 Is structural uncertainty a real problem in short-peptide AMPs?
       (P1-9 数据分析)
  4.3 Direct diagnosis of contamination from low-confidence regions
       (P0-1 直接扰动诊断)
  4.4 Main comparison with SOTA on two benchmarks
       (P0-3-B + P1-7 双 B 主结果)
  4.5 Ablation and mechanism validation
       (P0-3-A 双 B 消融 + P1-1 顺序 + P1-2 gate + P1-3 SGFN)
  4.6 Robustness across structural quality and predictors
       (P0-3-C robustness + P1-4 多信号 + P2-3 OmegaFold)
  4.7 Stronger uncertainty-aware baselines
       (P0-2)
  4.8 Transferability to related short-peptide tasks
       (P2-1, P2-2)
  4.9 (Optional) Reliability calibration
       (P2-4)
```

---

# 实验执行约束(全局)

## 统一性
- **5 seeds**:{37, 42, 123, 456, 789}
- **数据**:B1 来自 sAMPpred-GAT 划分,B2 来自 SSFGM-Model 划分
- **结构**:主实验用 ESMFold,robustness 实验用 AlphaFold2
- **环境**:必须用 `sucf_run` conda env(`/home/fyh0106/miniconda3/envs/sucf_run/bin/python`),
  默认 python 没有 mamba-ssm
- **GPU**:cuda:0 / cuda:3
- **训练超参**:遵循 `configs/training_config.yaml`,seed-level 配置在 `configs/exp_phase2_*.yaml`

## 每个实验都要回答
- 它在论文里回答哪个问题(对应章节号)
- 缺它会让论证链哪一环断掉

## 每个实验都要保存
- 原始结果表(JSON)
- 作图脚本
- 统计检验脚本
- 关键结论摘要(1 段话)
- 配置文件(seed / 超参)

## 命名规范
- 输出目录:`outputs/<phase>_<experiment_name>/seed_<n>/`
- 日志:`outputs/<experiment_name>/<seed>.log`

---

# 当前状态(2026-04-30)

## 已有代码
- `train_sucf.py`:主训练脚本
- `train_clean_ablation.py`:消融训练脚本(配 `models/sucf_clean_ablation.py`)
- `ablation_complete.py`:消融模型库(被多处引用)
- `run_clean_ablation.sh`:消融实验运行器
- `experiments/`:
  - `phase1_contamination_diagnosis/`(P0-1 雏形)
  - `phase2_ablation/`(P0-3-A)
  - `phase3_robustness/`(P0-3-C)
  - `phase4_mechanism_analysis/`(P1-2 / P1-3)
  - `contamination_diagnosis.py`(P0-1 主入口)
  - `pollution_diagnosis.py`
  - `calibration_order_experiment.py`(P1-1)
  - `analyze_plddt_gate_behavior.py`(P1-2)

## 已有结果(`outputs/`)
- `clean_ablation/`(14GB,4-29):双 B 消融最新结果,需校核入表
- `calibration_order/`(1.3GB,4-29):P1-1 部分结果

## 待补
- P0-1 直接扰动诊断:框架已搭(`contamination_diagnosis.py`),需要扩展扰动种类、对接对照模型
- P0-2 强 baseline 群:**完全空白**,优先级最高
- P0-3-B B2 上 SOTA 比较:**完全空白**
- P0-4 重画方法图:**未开始**
- 其他 P1 / P2 大部分待做

---

# 一句话核心
> **先把论文证据链补成:问题真实存在 → 污染可直接诊断 → 方法能缓解 → 优势可泛化。**
> 只要这条链完整,后续投稿都比现在更稳。
