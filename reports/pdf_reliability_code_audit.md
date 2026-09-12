# PDF reliability code audit

审计基线：`/data/projects/PDF`，commit `4eb619dca59a922fd28bc0891e8c3066e7186841`。
实际阅读代码后、模型修改前完成。实验仅使用一个新 worktree
`/data/projects/PDF-worktrees/pdf-reliability-ablation` / branch `pdf-reliability-ablation`。

## 实际执行路径

| 部分 | 代码证据（基线行号） | 实际行为 |
|---|---|---|
| entry / backbone | clip_finetune.py:89–111 | 硬编码 CLIP ViT-B/16，150 train IDs；不是 config 中的 resnet50；384×128，stride16 |
| image encoder | models/clip_model.py:449, 1200 | VisionTransformer 返回 projected tokens x 和 z；使用 x 的 CLS，512 维；BN 后分类 |
| text encoder | models/clip_model.py:777, 1052 | token embedding、positional embedding、CLIP causal transformer、LN、text projection；整个原始 text encoder 冻结，融合层可训练；返回全部 77×512 tokens |
| prompt-guided extractor | models/clip_model.py:797, 934, 1218 | text tokens → 现有 1-layer cross_modal_transformer → cross_attn(query=text,key=value=visual tokens) → caption EOT index → 共享 bottleneck_proj |
| decoupling | train.py:100–120 | f−a·com_proj 和 f−a_pos·com_proj；两个独立 per-sample Gaussian 系数 N(0.5,0.5²)，不是本轮 reliability alpha |
| training forward | train.py:89–120; models/clip_model.py:1210 | image + caption；分类分支与文本引导分支共享视觉 tokens 和 BN；返回 cls_score_proj、feat_proj、com_proj |
| inference | models/clip_model.py:1246; test.py:425,530 | image-only，只返回 BN 后 visual CLS 512维；原图+水平翻转后 L2 normalize；无 caption / semantic cache 依赖 |
| losses | train.py:96–120; losses/__init__.py:24 | label-smoothed CE + 已有 supervised contrastive pair loss + 0.5×OPL。TripletLoss 被实例化但没有调用，不能称为使用 triplet。CAL/clothes criterion 被构造但不参与总 loss |
| PRCC indexing | data/datasets/prcc.py:89–137 | train/person/*.jpg；目录名是原始 person_id，排序后映射为训练 label；A/B 同衣、C 异衣；tuple=(absolute image path,relabelled pid,camid,clothes_id) |
| caption lookup | data/dataset_loader.py:54–116 | absolute path / repo relative / data/prcc/rgb/train/person/file.jpg 查找；取 list 第0项；缺失抛 KeyError |

## 原文是什么

`data/captions/prcc.json` 中是逐图自然语言整句，例如
`A man wearing a white shirt and tan shorts.`，常含衣物、性别，偶尔背景信息。
不是 attribute-wise representation；未发现本轮四属性的独立 embedding、reliability 或额外 observation 机制。
PromptLearner 定义存在但没有在当前 active forward 中使用。

## 最小插入点及拟采用的静态 gating

保持原 caption（包括 clothing）及其编码处理，使用同一个已冻结 CLIP text encoder
为四种 identity attribute 的非 unknown 值编码固定短语，取 EOT embedding；作为输入预处理表缓存，
不新建 encoder、独立模型 branch 或参数。固定短语为 `A person with {attribute}: {value}.`。
这些是 CLIP 输入序列化，不修改已经完成的 Qwen P2 prompt / ontology / normalization。

仅对这四属性计算 `s=sum(r*e)/(sum(r)+1e-6)`；未知值直接零贡献，根本不编码 unknown 类别。
在训练的原 caption token features 的 EOT 位置做 `t[EOT] += s`，然后进入原 combine。
四种 mode 使用同一个位置；没有新 attention、新 loss、learned gate 或可训练参数。
原 caption 的 clothing 信息仍保留原路径且不加 reliability；不把 clothing 纳入 identity aggregation。
所有属性都无证据时 s=0，精确回到原 caption guidance。
这是 whole-caption 路径的 additive static gate，不声称与整句重编码数学等价。
推理 forward 的代码和 512 维输出均无需改变。

注意：原 com_proj 用于被减去的成分，identity guidance 加入后可能改变 decoupling 的效果；
因此收益必须由实验决定，不能据此预设 V1 或 V4 更优。

## 已发现的基线问题和公平性约束

1. 原 OPL `clothes_ids` 为一维，`eq(clothes_ids,clothes_ids.t())` 得到全 true 向量，
   不是 NxN clothes mask；sim_loss 为零，orth_loss 分母是 N 而不是 N²。
   双卡还遗漏 gather clothes_ids，可能产生 shape mismatch。
   本轮不修改 OPL；任何修正需要另开统一的 baseline family，不能混入 reliability 消融。
2. `/tmp/PDF_corrected_2gpu` 有未提交的 OPL clothes_ids gather 修补。
   旧 `/data/outputs/PDF/repro_corrected_2gpu` 与当前工作树不能仅凭名称视为相同实现。
3. 旧 canonical protocol 是 2×32/global64；本轮独立单卡进程拟使用 batch64/world1，
   BN 统计及采样拓扑不同，V0 必须在同一新协议下重跑，旧结果不复用。
4. 当前训练无条件 autocast / GradScaler；config.TRAIN.AMP=False 不能作为实际禁用 AMP 的证据。
5. 原 scheduler 每 epoch step，warmup_iters=10 实际是10个 epoch step，协议文档“10 iterations”不准确。
6. PRCC 当前 evaluation 是全 gallery multi-shot：A gallery，B same query，C diff query；
   dataset.gallery_idx 虽然生成但此 evaluator 不使用。返回 diff Rank1，日志输出全部 CMC/mAP。
7. 主结果固定 final epoch50；best test Diff Rank1 仅作辅助报告，统一 earliest tie，不能按每组最佳测试结果调参。

实际设备是两张 PPU-ZW810E，各98,304 MiB；PDF环境 torch CUDA接口可见2卡。
系统 PATH 中无 nvidia-smi，但 `/usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi` 可正常使用。

本轮用户的一个 branch/worktree、四 configs、独立单卡进程及指定输出目录要求，
优先于旧 EXPERIMENT_PROTOCOL.md 的逐消融 worktree / 单实验占双卡规则。
这里只执行预检，正式训练必须等待本轮用户确认。
