# PDF: Prompt-guided Decoupled Feature Learning for Cloth-Changing Person Re-identification

Official PyTorch implementation of the paper **"PDF: Prompt-guided Decoupled Feature Learning for Cloth-Changing Person Re-identification"**.

## 1. Installation


### Environment Setup
```bash
# Create a conda environment
conda create -n pdf_final python=3.9 -y
conda activate pdf_final

# Install PyTorch and dependencies
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 "numpy<2.0"
pip install yacs timm==0.5.4 scikit-image tqdm ftfy regex matplotlib h5py
```

## 2. Prepare Datasets

Download the cloth-changing ReID datasets into 'data' folder. The directory structure should look like this:

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

> **Note:** Please ensure the root path in your configuration files (`configs/*.yml`) points to your `data/` folder.

## 3. Training

We provide a convenient shell script to start the training process. This script handles the prompt-guided learning and feature decoupling.

To train the model on the specified dataset (e.g., LTCC), run:

```bash
# Run training
sh train.sh
```

*You can modify `train.sh` to switch between different datasets or hyper-parameters by changing the `--config_file` argument.*



## Acknowledgement
Our code is built upon [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID). We thank the authors for their great work.

---
