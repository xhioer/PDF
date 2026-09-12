# PDF + Reliability：预检交付

当前结论：步骤1–4通过；步骤5在首个 batch 的反向传播发现非有限梯度，按用户要求停止；
步骤6尚未启动。**不满足正式训练条件，V1–V4 长训练均未启动。**

工作树：`/data/projects/PDF-worktrees/pdf-reliability-ablation`。
分支：`pdf-reliability-ablation`。
训练代码及 smoke commit：`2bfba2f50f422512a82a8000665e7e9fc386c1dc`。
原 `/data/projects/PDF` 及已有其他 worktree 均未修改。

## 实际 data flow 与插入点

见 [完整代码审计](pdf_reliability_code_audit.md)。

图像 → CLIP ViT-B/16 projected tokens → visual CLS → BN → ID classification。
旧逐图整句 caption → 冻结 CLIP text encoder → 77×512 tokens →
现有 cross_modal_transformer → text-to-visual cross-attention → EOT → 共享 BN → com_proj。
training 用 `f-a*com_proj` 两个随机视图计算原 pair loss，并保留原 OPL。
inference 是纯图像 BN 后512维特征，配合 horizontal-flip TTA、L2 normalization。

P2只读取 gender、hair_color、hair_length、body_build 及 confidence；
用同一冻结 text encoder 编码有效属性短语，形成不可训练的输入 embedding 表。
`s=sum(r*e)/(sum(r)+1e-6)`，然后在原 token feature 的 EOT 处 `t[EOT]+=s`。
原 caption 的 clothing 保持原路径；不把 clothing 放入 identity aggregation；不读取 observation 字段参与模型。
全零证据得到 s=0；unknown 没有类别 embedding。

这测量的是“原 PDF caption guidance + P2 identity EOT additive guidance”的 reliability 消融，
不是用完整 P2 字段替换原 caption，也不声称 additive gate 等价于整句重新编码。
最终收益只能由相同协议的 V0–V4 实验确定。

## 数据和 sanity

[Cache validation](prcc_semantic_cache_validation.json)：
17,896 train images / 17,896 records / 17,896 unique matches；
missing=0、duplicate=0、extra=0、invalid schema=0、invalid JSON=0、invalid metadata=0。
实际调用 PDF PRCC loader，检查完整路径与原始 person_id（不是 relabelled ID）。
保留完整 normalized description、raw_model_output、image_path、person_id；未补 unknown。
cache SHA256：`cd0aded0585c6af9f65451a7a05cb7a088601057600533fb2efd151d1275f20d`。

[Sanity JSON](reliability_sanity_check.json) 包含每个 mode/attribute 的 mean、std、min、max、zero_rate、unique_values，
同时包含100个实际图像 batch 和全 cache 的统计。抽样 batch64×100，共6,400张不同图像，seed0。

以下为抽样 mean，顺序固定为 gender / hair_color / hair_length / body_build：

| mode | gender | hair_color | hair_length | body_build |
|---|---:|---:|---:|---:|
| none | 1.000000 | 0.987344 | 0.996719 | 1.000000 |
| attribute | 0.975300 | 0.881204 | 0.842526 | 0.773100 |
| confidence | 0.900547 | 0.639219 | 0.800859 | 0.532500 |
| joint | 0.878303 | 0.570503 | 0.676966 | 0.411676 |

none/attribute zero rates：0%、1.265625%、0.328125%、0%。
confidence/joint zero rates：0%、4.046875%、1.28125%、0%。
所有观测值均符合固定1、alpha、beta、alpha×beta；JSON保留float32表示误差。
四模式任意两组权重张量不相同；四config解析后除 mode 外完全相同。

[真实 CLIP 集成验证](reliability_integration_check.json)：
新增 trainable parameters=0；原 trainable parameters=90,469,888。
inference shape=[2,512]，开启/关闭 guidance 时 image-only inference 逐值相同；
全零证据与原 guidance 逐值相同；四种 mode 的实际融合输出两两不同。

## 冻结训练协议与 V0

四份配置为 `configs/v1_p2_none.py`、`v2_p2_attribute.py`、`v3_p2_confidence.py`、`v4_p2_joint.py`；
共用 `configs/reliability_common.py`，唯一变化为 SEMANTIC_RELIABILITY_MODE。
输出路径从 mode 派生，四个正式输出目录已创建且为空。

共同协议：PRCC原split、ViT-B/16、本地同一个CLIP checkpoint、384×128、Adam、lr=3.5e-7、
weight decay=5e-4、50 epochs、warmup10个scheduler steps（实际每epoch一步）、milestones20/40、
gamma0.1、batch64/world1、NUM_INSTANCES8、workers4、seed0、AMP、
原resize/flip/erasing、原CE/pair/OPL，eval每5epoch。
Triplet loss未被原训练调用，因此日志标为0/inactive，不新增triplet或semantic loss。
主结果统一 final epoch50，辅助记录 earliest best test Diff Rank1 和 best epoch。
不根据中期测试指标调参。

V0将使用同一个 runner 的 `--v0`，只关闭 P2输入。
旧双卡batch32×2 baseline在BN统计、采样拓扑和未提交OPL修补方面不满足同协议要求，**不复用，必须重跑V0**。
原 OPL 的一维 clothes mask 问题已明确记入审计；本轮 losses目录与基线逐字节相同，未擅自修正。
因此后续结果应注明是当前实现家族，不能当作已纠正OPL的 paper-faithful family。

## Smoke 失败与诊断

[资源预检状态](reliability_resource_test.json)。
V4 joint / Device0 / batch64 / world1 / 目标300秒；epoch1 batch1非有限梯度，
在optimizer.step前停止，成功optimizer updates=0。
未得到有效的稳态显存或速度数据，不能用初始化显存替代训练显存。
单卡双进程未启动，因此显存、吞吐、swap及是否可同时跑四组均未通过验证。

[AMP诊断](reliability_amp_diagnostic.json) 使用同一真实图像batch，重置RNG与BN；所有诊断optimizer updates=0：

| 输入 | 反向 scale | loss | 非有限梯度参数张量数 |
|---|---:|---:|---:|
| V0 | 65536 | 8.882695 | 32 |
| V0 | 1 | 8.882695 | 0 |
| V4 | 65536 | 8.886599 | 32 |
| V4 | 1 | 8.886599 | 0 |

证据支持“默认初始 AMP loss scale 过大导致首次反向溢出”，而非V4独有的问题。
原 GradScaler 通常会跳过overflow step并动态降低scale；本轮采用用户要求的严格no-NaN/Inf停止条件，
所以没有让失败实验自动跳过后继续，也没有修改初始化scale重跑。
scale1仅为定位原因的反向probe，不是训练结果，不能据此宣称完整训练稳定。

下一步建议：获得用户明确确认后，将V0–V4共享的AMP初始loss scale统一设为1.0，
重新固定协议/commit，先重做单进程300秒和双进程资源测试；仍需后续确认才能启动正式长训练。
这改变了已冻结的AMP配置，当前未实施。若还要修复原OPL，则需单独决定baseline family，不能暗中纳入本轮。

## 修改文件与交付能力

- `models/clip_model.py`：training-only EOT additive guidance，未修改image-only inference。
- `data/semantic_reliability.py`：固定常量、strict validation、统一r和parameter-free aggregation。
- `data/dataset_loader.py`、`data/__init__.py`：唯一映射、原caption保留、训练lookup payload。
- `configs/reliability_common.py`及四config：共享冻结协议。
- `train.py`：共享semantic输入、iteration metrics、非有限loss/gradient停止、smoke时间上限。
- `test.py`：原PRCC evaluator增加可选结构化完整SC/CC CMC/mAP返回，不改变metric算法。
- `tools/validate_semantic_cache.py`、`reliability_sanity.py`、`check_reliability_integration.py`：数据/权重/真实forward验证。
- `tools/run_reliability.py`：独立进程runner，默认smoke；config/git/log/metrics/stats/status及正式best/final checkpoint保存。
- `tools/diagnose_reliability_amp.py`：零optimizer-update诊断。
- `tools/collect_reliability_results.py`：正式epoch50完成后校验同commit/协议并输出五行消融表和六项差值。
- `.gitignore`：忽略新生成pyc和outputs；未删除已有tracked文件。

没有产生任何正式ReID指标；不生成填造数字的消融表，不对RQ1–RQ4下结论，也不能选择RQ5的A/B/C/D。
collector在不完整结果上会报错，而不是把smoke当作正式训练。
