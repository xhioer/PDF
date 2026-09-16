# PDF Identity-Semantic Preservation V2 overnight report

生成时间（UTC）：2026-09-13T05:49:01Z
实现 commit：`878ac96afe4a7772a324586540d86c7ed2ac277e`
输出根目录：`outputs/pdf_identity_preservation_v2`

## 结论

**自动分类：B**。identity-preservation 有限正向证据：N09 matched Diff R1 Δ=0.1599±0.1879 pp、Diff mAP Δ=-0.0124±0.0083 pp；N16 joint matched Diff mAP Δ=0.0112±0.0110 pp、R1 Δ=0.0094±0.0326 pp。整体支持继续验证，但不支持广泛或明显的 reliability 增益。

继续 identity-semantic preservation 方向：**WEAK YES**。

最有希望配置（分别代表 R1 机制候选与 reliability/mAP 候选）：
- `N09`：Delta Diff R1 mean=0.1599 pp，Delta Diff mAP mean=-0.0124 pp；matched R1 deltas=0.3669, 0.1129, 0.0000。
- `N16`：joint reliability；Delta Diff R1 mean=0.0094±0.0326 pp，Delta Diff mAP mean=0.0112±0.0110 pp；Same R1=99.8881±0.0149。

## Implementation audit

- tensor audit：见 [identity_preservation_tensor_audit.md](identity_preservation_tensor_audit.md)。
- semantic cache：固定 full PRCC train 17,896/17,896；属性固定为 gender、hair_color、hair_length、body_build；epsilon=1e-6；未知属性权重为0；all-unknown sample 不进入 semantic loss 分母。
- 原 caption clothing branch 保留；所有 V2 run 的 EOT additive guidance 为 OFF；没有新增 projection、attention、encoder、backbone、observation gate 或 OPL/triplet 修改。
- gradient sanity：见 [identity_preservation_gradient_sanity.json](identity_preservation_gradient_sanity.json)；passed=True，四个 loss finite=True，image-only bitwise unchanged=True，visual trainable tensors=152，trainable parameter count before/after=90469888/90469888。

## 27 个实验完成状态

详表见 [identity_preservation_run_status.csv](identity_preservation_run_status.csv)。完成数：27/27；失败数：0；未启动数：0。

## Seed0 全表

见 [identity_preservation_seed0_ablation.csv](identity_preservation_seed0_ablation.csv)。主指标为 epoch50 final；best test checkpoint 只作为 run 内辅助诊断。

| Variant | Mechanism | Reliability | λraw | λpres | λexcl | λrank | margin | Diff R1 | Diff mAP | Same R1 | Same mAP | cos_res | cos_com | margin diagnostic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N00 | Original PDF / control | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 64.4087 | 62.1613 | 99.8709 | 98.1416 | 0.002663 | -0.000466 | 0.003129 |
| N01 | Raw Visual Alignment | none | 0.0500 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 64.5780 | 62.1602 | 99.8709 | 98.1371 | 0.002789 | -0.000472 | 0.003261 |
| N02 | Residual Identity Preservation | none | 0.0000 | 0.0200 | 0.0000 | 0.0000 | 0.1000 | 64.4651 | 62.1548 | 99.8709 | 98.1423 | 0.002686 | -0.000477 | 0.003162 |
| N03 | Residual Identity Preservation | none | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 0.1000 | 64.4087 | 62.1272 | 99.8709 | 98.1261 | 0.002769 | -0.000527 | 0.003295 |
| N04 | Residual Identity Preservation | none | 0.0000 | 0.1000 | 0.0000 | 0.0000 | 0.1000 | 64.3805 | 62.1461 | 99.8709 | 98.1471 | 0.002888 | -0.000601 | 0.003488 |
| N05 | Identity Leakage Exclusion | none | 0.0000 | 0.0000 | 0.0200 | 0.0000 | 0.1000 | 64.5498 | 62.1803 | 99.8709 | 98.1440 | 0.002664 | -0.000469 | 0.003132 |
| N06 | Identity Leakage Exclusion | none | 0.0000 | 0.0000 | 0.0500 | 0.0000 | 0.1000 | 64.5498 | 62.1612 | 99.8709 | 98.1302 | 0.002648 | -0.000453 | 0.003101 |
| N07 | Identity Leakage Exclusion | none | 0.0000 | 0.0000 | 0.1000 | 0.0000 | 0.1000 | 64.2958 | 62.1363 | 99.8709 | 98.1366 | 0.002643 | -0.000452 | 0.003095 |
| N08 | Relative Semantic Ranking | none | 0.0000 | 0.0000 | 0.0000 | 0.0200 | 0.1000 | 64.2111 | 62.1371 | 99.8709 | 98.1430 | 0.002703 | -0.000573 | 0.003276 |
| N09 | Relative Semantic Ranking | none | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.1000 | 64.7756 | 62.1583 | 99.8709 | 98.1339 | 0.002814 | -0.000761 | 0.003575 |
| N10 | Relative Semantic Ranking | none | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.2000 | 64.2676 | 62.1389 | 99.8709 | 98.1382 | 0.002844 | -0.000802 | 0.003646 |
| N11 | Relative Semantic Ranking | none | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.2000 | 64.4087 | 62.1304 | 99.8709 | 98.1416 | 0.003042 | -0.001150 | 0.004192 |
| N12 | Preservation + Exclusion | none | 0.0000 | 0.0500 | 0.0200 | 0.0000 | 0.1000 | 64.3522 | 62.0822 | 99.8709 | 98.1422 | 0.002777 | -0.000541 | 0.003319 |
| N13 | Preservation + Exclusion | none | 0.0000 | 0.0500 | 0.0500 | 0.0000 | 0.1000 | 64.4651 | 62.1683 | 99.8709 | 98.1394 | 0.002775 | -0.000519 | 0.003295 |
| N14 | Relative Semantic Ranking + Attribute Reliability | attribute | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.1000 | 64.4087 | 62.1137 | 99.8709 | 98.1478 | 0.003013 | -0.000749 | 0.003761 |
| N15 | Relative Semantic Ranking + Confidence Reliability | confidence | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.1000 | 64.5498 | 62.1646 | 99.8709 | 98.1437 | 0.003361 | -0.000804 | 0.004164 |
| N16 | Relative Semantic Ranking + Joint Reliability | joint | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.1000 | 64.4369 | 62.1699 | 99.8709 | 98.1363 | 0.003596 | -0.000807 | 0.004403 |

## Different Clothes 排名（seed0 epoch50 final，按 R1）

1. `N09`：R1=64.7756，mAP=62.1583；Delta R1=0.3669，Delta mAP=-0.0031。
2. `N01`：R1=64.5780，mAP=62.1602；Delta R1=0.1693，Delta mAP=-0.0011。
3. `N05`：R1=64.5498，mAP=62.1803；Delta R1=0.1411，Delta mAP=0.0190。
4. `N06`：R1=64.5498，mAP=62.1612；Delta R1=0.1411，Delta mAP=-0.0001。
5. `N15`：R1=64.5498，mAP=62.1646；Delta R1=0.1411，Delta mAP=0.0033。
6. `N02`：R1=64.4651，mAP=62.1548；Delta R1=0.0564，Delta mAP=-0.0065。
7. `N13`：R1=64.4651，mAP=62.1683；Delta R1=0.0564，Delta mAP=0.0070。
8. `N16`：R1=64.4369，mAP=62.1699；Delta R1=0.0282，Delta mAP=0.0086。
9. `N00`：R1=64.4087，mAP=62.1613；Delta R1=0.0000，Delta mAP=0.0000。
10. `N03`：R1=64.4087，mAP=62.1272；Delta R1=0.0000，Delta mAP=-0.0341。
11. `N11`：R1=64.4087，mAP=62.1304；Delta R1=0.0000，Delta mAP=-0.0309。
12. `N14`：R1=64.4087，mAP=62.1137；Delta R1=0.0000，Delta mAP=-0.0476。
13. `N04`：R1=64.3805，mAP=62.1461；Delta R1=-0.0282，Delta mAP=-0.0152。
14. `N12`：R1=64.3522，mAP=62.0822；Delta R1=-0.0564，Delta mAP=-0.0791。
15. `N07`：R1=64.2958，mAP=62.1363；Delta R1=-0.1129，Delta mAP=-0.0250。
16. `N10`：R1=64.2676，mAP=62.1389；Delta R1=-0.1411，Delta mAP=-0.0224。
17. `N08`：R1=64.2111，mAP=62.1371；Delta R1=-0.1976，Delta mAP=-0.0242。

## 3-seed matched 结果

见 [identity_preservation_multiseed.csv](identity_preservation_multiseed.csv) 和 per-seed 明细。所有 Delta 都是同 seed 的 `M_variant,s - M_N00,s`。

| Variant | Reliability | Diff R1 mean±std | Diff mAP mean±std | Same R1 mean±std | Same mAP mean±std | matched ΔDiff R1 mean±std | matched ΔDiff mAP mean±std |
|---|---|---:|---:|---:|---:|---:|---:|
| `N00` | none | 65.3495±1.0965 | 62.3256±0.1963 | 99.8709±0.0000 | 98.1325±0.1045 | 0.0000±0.0000 | 0.0000±0.0000 |
| `N09` | none | 65.5095±0.9288 | 62.3132±0.1890 | 99.8537±0.0298 | 98.1238±0.1113 | 0.1599±0.1879 | -0.0124±0.0083 |
| `N14` | attribute | 65.3966±1.0489 | 62.3118±0.2254 | 99.8537±0.0298 | 98.1301±0.1098 | 0.0470±0.1334 | -0.0138±0.0313 |
| `N15` | confidence | 65.4154±1.0892 | 62.3100±0.1947 | 99.8623±0.0394 | 98.1313±0.1035 | 0.0659±0.0862 | -0.0156±0.0211 |
| `N16` | joint | 65.3589±1.0656 | 62.3368±0.1907 | 99.8881±0.0149 | 98.1295±0.1086 | 0.0094±0.0326 | 0.0112±0.0110 |

## Same Clothes 稳定性

- `N00`：Same R1 99.8709±0.0000，Same mAP 98.1325±0.1045。
- `N16`：Same R1 99.8881±0.0149，Same mAP 98.1295±0.1086。
- `N09`：Same R1 99.8537±0.0298，Same mAP 98.1238±0.1113。
- `N14`：Same R1 99.8537±0.0298，Same mAP 98.1301±0.1098。
- `N15`：Same R1 99.8623±0.0394，Same mAP 98.1313±0.1035。

## Semantic cosine / margin diagnostics

每个 run 的 `epoch_diagnostics.csv` 记录 `cos_raw_sem`、`cos_res1_sem`、`cos_res2_sem`、`cos_com_sem`、两条 residual-minus-com margin、四个 loss 和 semantic_valid_rate；seed0 表汇总 epoch50 的 residual 平均 cosine、com cosine 和平均 margin。

## 机制问题自动回答

- Q1 Raw alignment：N01 相对 N00 为 Diff R1 0.1693 pp、Diff mAP -0.0011 pp；结论：只有很小的 R1 单 seed 增益，mAP 基本 neutral/略降，不能认为 direct alignment 已有效。
- Q2 Residual preservation：N02–N04 的最佳 Diff R1=64.4651、Diff mAP=62.1548，均未超过 N01 的 R1=64.5780、mAP=62.1602；结论：本轮三个 preservation 权重没有优于 raw alignment，虽 residual cosine/margin 随权重上升但未转化为 ReID 收益。
- Q3 Leakage exclusion：N05/N06/N07 的 seed0 Diff R1 Δ分别为 0.1411/0.1411/-0.1129 pp、mAP Δ分别为 0.0190/-0.0001/-0.0250 pp；`cos_com_sem` 约在 -0.000469 到 -0.000452，没有随 lambda 单调下降；结论：低权重 N05 有小幅正向，但 exclusion 尚无稳定证据。
- Q4 Relative ranking：N08/N09/N10/N11 的 seed0 Diff R1 Δ为 -0.1976/0.3669/-0.1411/0.0000 pp，mAP Δ为 -0.0242/-0.0031/-0.0224/-0.0309 pp；margin diagnostic 均为正但性能不随 lambda/margin 单调，N09 的 3-seed matched R1 为 0.1599±0.1879 pp、mAP 为 -0.0124±0.0083 pp；结论：ranking 改变了 decomposition，但“更稳定”尚未被充分支持。
- Q5 Preservation + exclusion：N12 为 Diff R1 Δ=-0.0564、mAP Δ=-0.0791，N13 为 Diff R1 Δ=0.0564、mAP Δ=0.0070；结论：只有较高 exclusion 权重的 N13 显示弱的 seed0 互补迹象，N12 反而受损，不能确认普遍互补。
- Q6 Reliability：matched 3-seed Delta Diff R1/mAP 分别为 N09=0.1599/-0.0124、N14=0.0470/-0.0138、N15=0.0659/-0.0156、N16=0.0094/0.0112 pp；结论：attribute/confidence 没有稳定额外收益，joint 的 mAP 为正且更稳定，但 R1 仍接近 neutral。
- Q7 Multi-seed：N16 joint 的 matched Diff mAP deltas 为 0.0086, 0.0017, 0.0233，三个 seed 均为正；Diff R1 deltas 为 0.0282, 0.0282, -0.0282，仅小幅正/负摆动；Same R1 为 99.8881±0.0149 且是五个关键配置中波动最小；结论：matched evidence 最支持 joint 用于 mAP/Same Clothes 稳定性，不支持明显 Diff R1 提升。详见 [identity_preservation_multiseed_per_seed.csv](identity_preservation_multiseed_per_seed.csv)。

## 与旧 V0–V4 比较

- 已知旧 Original PDF V0：Different Clothes R1=64.4087，mAP=62.1613。
- 旧 V0–V4 CSV：/data/projects/PDF-worktrees/pdf-reliability-ablation/reports/pdf_reliability_ablation.csv。若存在，将在自动分类中作为旧 EOT additive family 的参考；否则不虚构缺失数字。

| Variant | Semantic | Attribute reliability | Confidence reliability | Diff R1 | Diff mAP | Same R1 | Same mAP |
|---|---|---|---|---:|---:|---:|---:|
| V0 | Original PDF | No | No | 64.4087 | 62.1613 | 99.8709 | 98.1416 |
| V1 | P2 | No | No | 64.4651 | 62.1635 | 99.8709 | 98.1472 |
| V2 | P2 | Yes | No | 64.6909 | 62.1725 | 99.8709 | 98.1443 |
| V3 | P2 | No | Yes | 64.4651 | 62.2009 | 99.8709 | 98.1500 |
| V4 | P2 | Yes | Yes | 64.2393 | 62.1984 | 99.8709 | 98.1487 |

- 旧单 seed 中最佳 Diff R1 为 V2=64.6909，最佳 Diff mAP 为 V3=62.2009；新 seed0 中 N09 的 Diff R1=64.7756（比旧 V2 高 0.0847 pp），但 Diff mAP=62.1583（比旧 V3 低 -0.0427 pp）。这是单 seed 对比，不能替代 matched multi-seed 结论。
- 新 V2 所有主结果统一为 epoch50 final；不能用各组 best test epoch 替换主结果。

## 最终推荐

- 配置推荐 1：`N09`（none + rank λ=0.05、margin=0.10）作为当前最强的 Diff R1 机制候选，但必须接受其 mAP 没有提升、matched R1 效应小于 seed std 的事实。
- 配置推荐 2：`N16`（joint + rank λ=0.05、margin=0.10）作为 reliability 候选；三个 matched seed 的 Diff mAP delta 全为正，且 Same R1 波动最小，但 Diff R1 基本 neutral。
- 最推荐下一步：冻结 `N16` 做更大 seed 数的确认，并按属性/样本记录 decomposition change；若目标只看 Different Clothes R1，则并行保留 `N09`，不要再做 test-adaptive lambda 搜索。若确认效应仍接近 seed variance，应转向重新设计四属性 semantic representation。
判断原则：如果 ReID 不升但 residual/com cosine 和 margin 按预期变化，应把它作为 decomposition 机制证据，而不是伪造性能收益。

不成功实验、失败状态和未启动任务均保留在 status CSV、各 run 目录及 queue status 中。
