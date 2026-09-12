from pathlib import Path
from configs.default_img import _C
from data.semantic_reliability import CACHE_PATH, MODES


def build_config(mode):
    if mode not in MODES:
        raise ValueError('Invalid SEMANTIC_RELIABILITY_MODE')
    config = _C.clone()
    config.merge_from_file(str(Path(__file__).parent / 'prcc_2gpu.yaml'))
    config.DATA.ROOT = '/data/datasets/PRCC'
    config.DATA.CAPTION_PATH = str(Path(__file__).resolve().parents[1] / 'data/captions/prcc.json')
    config.DATA.SEMANTIC_CACHE = CACHE_PATH
    config.DATA.TRAIN_BATCH = 64  # original effective batch64, now world_size=1
    config.DATA.NUM_WORKERS = 4
    config.SEMANTIC_RELIABILITY_MODE = mode
    config.TRAIN.AMP = True  # original train.py actually always used AMP
    config.OUTPUT = str(Path(__file__).resolve().parents[1] / 'outputs/pdf_reliability_ablation')
    config.TAG = 'pdf-reliability-ablation'
    config.freeze()
    return config
