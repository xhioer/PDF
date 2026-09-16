# PDF 实验协议

本文档是 PRCC 复现与消融实验的强制检查清单。

## 固定路径

| 项目 | 必须使用的路径 |
| --- | --- |
| 项目目录 | `/data/projects/PDF` |
| Python 环境 | `/data/envs/PDF` |
| 数据集根目录参数 | `/data/datasets/PRCC` |
| PRCC 图像目录 | `/data/datasets/PRCC/prcc/rgb` |
| 默认训练 caption | `/data/projects/PDF/data/captions/prcc.json` |
| 实验输出目录 | `/data/outputs/PDF` |
| CLIP ViT-B/16 权重 | `/root/.cache/clip/ViT-B-16.pt` |

新的实验不得使用 `/home/outputs`。CLIP 权重缓存是输入依赖，不是实验输出。

caption 文件可配置。将 `PDF_CAPTION_FILE` 设为 `/data` 下的绝对路径；运行器会把该路径传给 PRCC loader 并记录其 SHA-256 哈希。默认文件为 `/data/projects/PDF/data/captions/prcc.json`。

## 运行环境

- 使用 `/data/envs/PDF/bin/python`；消融实验中不得安装或升级 `torch`、`torchvision`。
- 在导入 PyTorch 前设置 `LD_LIBRARY_PATH=/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64`。
- 使用 PPU-ZW810E 的 0、1 号卡，并设置 `CUDA_VISIBLE_DEVICES=0,1`。
- 使用两个分布式进程，每张卡 batch size 为 32，全局 batch size 为 64。
- 一次只运行一个实验；不得与其他训练任务共享这两张卡。
- 除非整个实验矩阵同步调整，否则保持 `MKL_NUM_THREADS=1` 与 `OMP_NUM_THREADS=1`。

标准命令：

```bash
cd /data/projects/PDF
bash train.sh
```

运行消融实验时，只修改配置和标签：

```bash
PDF_CFG=configs/ablations/no_opl.yaml PDF_TAG=no_opl bash train.sh
```

使用不同 caption 版本时：

```bash
PDF_CAPTION_FILE=/data/captions/prcc_v2.json PDF_TAG=caption-v2 bash train.sh
```

运行器会拒绝 `/data/outputs/PDF` 之外的输出路径，并在每个运行目录中写入 `run_manifest.txt` 与 `git_diff.patch`。

## 数据集与评估

- 将数据集和配置的 caption 文件视为只读输入。
- 同一比较表中保持一致的 PRCC train/val/test split 和 caption 文件。
- PRCC 预期数量为：17,896 张训练图像、5,002 张验证图像、3,873 个同衣查询、3,543 个换衣查询、3,384 张 gallery 图像。
- 使用相同的 all-gallery multi-shot 评估协议、image-only inference、horizontal-flip test-time augmentation，以及 `test.py` 实现。
- 使用 ViT-B/16、输入尺寸 384x128、random horizontal flip、random erasing、Adam、学习率 3.5e-7、weight decay 5e-4、50 epochs、前 10 iterations warmup，以及在第 20、40 epoch 学习率衰减。

## 代码隔离

- 保持 `/data/projects/PDF` 为稳定基线；不要在不同运行之间反复编辑。
- 每个消融实验创建一个独立 Git worktree 和一个独立分支。
- 每次只修改一个模块或一个 loss；可能时将变更放入 `configs/ablations/<experiment_id>.yaml`。
- 将 official-code baseline 与 paper-faithful corrected baseline 作为不同实验家族维护；不得在同一张消融表中混用。
- 每次运行都记录 baseline commit、final commit、diff、配置与环境。

推荐目录结构：

```text
/data/projects/PDF-worktrees/
  ablation-no-opl/
  ablation-no-text-freeze/
  ablation-shared-alpha/

/data/outputs/PDF/ablations/
  official_baseline/seed_0/
  no_opl/seed_0/
  no_text_freeze/seed_0/
```

## 报告规范

- 后续实验报告、结果解释、图注和结论默认使用中文。为保证精确性，标准技术名词、指标标签、代码标识、文件路径和方法名可保留原文。
- 使用 seed 0 进行 smoke test；计算资源允许时，正式报告的消融实验使用 seeds 0、1、2。
- 以 epoch50 为主结果。best test Rank-1 应单独报告；不得只为每种方法选择不同的 test epoch 作为唯一结果。
- 报告 CC Rank-1、CC mAP、SC Rank-1 和 SC mAP；多 seed 时同时报告均值和标准差。
- 除非实现、数据访问、worker 数、有效 batch size 及 seed 其余均相同，否则不得比较单卡运行与双卡运行。

## 运行前检查清单

每次运行前执行：

```bash
test -x /data/envs/PDF/bin/python
test -d /data/datasets/PRCC/prcc/rgb/train
test -d /data/datasets/PRCC/prcc/rgb/val
test -d /data/datasets/PRCC/prcc/rgb/test
test -f "${PDF_CAPTION_FILE:-/data/projects/PDF/data/captions/prcc.json}"
test -d /data/outputs/PDF
rg -n '/home/outputs|/home/envs' train.sh configs data || true
```

最后一条命令不得返回新的 `/home` 输出或环境路径。
