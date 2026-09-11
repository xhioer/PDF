# PDF: Prompt-guided Decoupled Feature Learning for Cloth-Changing Person Re-identification

Official PyTorch implementation of the paper **"PDF: Prompt-guided Decoupled Feature Learning for Cloth-Changing Person Re-identification"**.

## 1. Installation


### Environment Setup

The reproduction machine already provides the required environment at `/data/envs/PDF`. Use its Python executable directly; do not install or upgrade `torch` or `torchvision` for reproduction or ablation runs.

```bash
/data/envs/PDF/bin/python --version
```

## 2. Prepare Datasets

For the current PRCC reproduction, use the read-only dataset at `/data/datasets/PRCC/prcc`. The default training captions are `data/captions/prcc.json`; the caption file can be changed per run without editing the loader.

The expected PRCC directory structure is:

```text
PDF/
└── data/
    ├── LTCC_ReID/
    │   ├── train/
    │   ├── test/
    │   └── query/
    ├── prcc/
    │   ├── rgb/
    │       ├── val/
    │       ├── test/
    │       └── train/
    ├── last/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── VC-Clothes/
        ├── train/
        ├── query/
        └── gallery/
```

> **Note:** The PRCC root passed to the training command is `/data/datasets/PRCC`, which contains the `prcc/` directory.

## 3. Training

The canonical `train.sh` runs the standard two-card PRCC experiment with PPU-ZW810E cards 0 and 1, global batch size 64, and writes all results to `/data/outputs/PDF`.

Run the standard experiment with:

```bash
bash train.sh
```

For ablations, use a separate Git worktree and pass only an ablation config and tag:

```bash
PDF_CFG=configs/ablations/no_opl.yaml PDF_TAG=no_opl bash train.sh
```

To use another caption file, set `PDF_CAPTION_FILE`:

```bash
PDF_CAPTION_FILE=/data/captions/prcc_v2.json PDF_TAG=caption-v2 bash train.sh
```

See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the mandatory paths, runtime settings, isolation rules, preflight checks, and reporting requirements.



## Acknowledgement
Our code is built upon [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID). We thank the authors for their great work.

---
