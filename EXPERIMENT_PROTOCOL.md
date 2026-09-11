# PDF Experiment Protocol

This document is the mandatory checklist for PRCC reproduction and ablation runs.

## Fixed Paths

| Item | Required path |
| --- | --- |
| Project | `/data/projects/PDF` |
| Python environment | `/data/envs/PDF` |
| Dataset root argument | `/data/datasets/PRCC` |
| PRCC images | `/data/datasets/PRCC/prcc/rgb` |
| Default training captions | `/data/projects/PDF/data/captions/prcc.json` |
| Experiment outputs | `/data/outputs/PDF` |
| CLIP ViT-B/16 weights | `/root/.cache/clip/ViT-B-16.pt` |

`/home/outputs` must not be used for new experiments. The CLIP weight cache is an input dependency, not an experiment output.

The caption file is configurable. Set `PDF_CAPTION_FILE` to an absolute path under `/data`; the runner passes that path to the PRCC loader and records its SHA-256 hash. The default is `/data/projects/PDF/data/captions/prcc.json`.

## Runtime

- Use `/data/envs/PDF/bin/python`; do not install or upgrade `torch` or `torchvision` for an ablation.
- Export `LD_LIBRARY_PATH=/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64` before importing PyTorch.
- Use PPU-ZW810E cards 0 and 1 with `CUDA_VISIBLE_DEVICES=0,1`.
- Use two distributed processes, batch32 per card, and global batch64.
- Run one experiment at a time; do not share the cards with another training job.
- Keep `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` unless the whole experiment matrix is changed together.

The canonical command is:

```bash
cd /data/projects/PDF
bash train.sh
```

For an ablation, change only the config and tag:

```bash
PDF_CFG=configs/ablations/no_opl.yaml PDF_TAG=no_opl bash train.sh
```

For a different caption version:

```bash
PDF_CAPTION_FILE=/data/captions/prcc_v2.json PDF_TAG=caption-v2 bash train.sh
```

The runner refuses output paths outside `/data/outputs/PDF` and writes `run_manifest.txt` and `git_diff.patch` into each run directory.

## Dataset and Evaluation

- Treat the dataset and the configured caption file as read-only inputs.
- Keep the same PRCC train/val/test split and caption file within one comparison table.
- Expected PRCC counts are 17,896 train images, 5,002 validation images, 3,873 same-clothes queries, 3,543 clothes-changing queries, and 3,384 gallery images.
- Use the same all-gallery multi-shot evaluation protocol, image-only inference, horizontal-flip test-time augmentation, and `test.py` implementation.
- Use ViT-B/16, input size 384x128, random horizontal flip, random erasing, Adam, learning rate 3.5e-7, weight decay 5e-4, 50 epochs, warmup for 10 iterations, and learning-rate decay at epochs 20 and 40.

## Code Isolation

- Keep `/data/projects/PDF` on a stable baseline; do not edit it repeatedly between runs.
- Create one Git worktree and one branch per ablation.
- Change one module or one loss at a time. Put the change in `configs/ablations/<experiment_id>.yaml` when possible.
- Keep the official-code baseline and the paper-faithful corrected baseline as separate experiment families; never mix them in one ablation table.
- Record the baseline commit, final commit, diff, configuration, and environment for every run.

Recommended layout:

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

## Reporting Rules

- Use seed 0 for a smoke test; use seeds 0, 1, and 2 for reported ablations when compute permits.
- Keep epoch50 as the primary result. Report best test Rank-1 separately; do not select a different test epoch for each method as the only result.
- Report CC Rank-1, CC mAP, SC Rank-1, and SC mAP, plus mean and standard deviation when multiple seeds are used.
- Do not compare a one-card run with a two-card run unless the implementation, data access, worker count, effective batch size, and seed are otherwise identical.

## Preflight Checklist

Before every run, verify:

```bash
test -x /data/envs/PDF/bin/python
test -d /data/datasets/PRCC/prcc/rgb/train
test -d /data/datasets/PRCC/prcc/rgb/val
test -d /data/datasets/PRCC/prcc/rgb/test
test -f "${PDF_CAPTION_FILE:-/data/projects/PDF/data/captions/prcc.json}"
test -d /data/outputs/PDF
rg -n '/home/outputs|/home/envs' train.sh configs data || true
```

The last command must return no new `/home` output or environment path.
