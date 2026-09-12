"""Frozen PRCC protocol and V2 identity-preservation config factory."""

from pathlib import Path

from yacs.config import CfgNode as CN

from configs.default_img import _C
from data.semantic_reliability import CACHE_PATH, MODES


ROOT = Path(__file__).resolve().parents[1]


def build_config(spec, seed, output):
    config = _C.clone()
    config.merge_from_file(str(Path(__file__).parent / 'prcc_2gpu.yaml'))
    config.DATA.ROOT = '/data/datasets/PRCC'
    config.DATA.CAPTION_PATH = str(ROOT / 'data/captions/prcc.json')
    config.DATA.SEMANTIC_CACHE = CACHE_PATH
    config.DATA.TRAIN_BATCH = 64
    config.DATA.NUM_WORKERS = 4
    config.SEMANTIC_RELIABILITY_MODE = spec['reliability']
    if spec['reliability'] not in MODES:
        raise ValueError('Invalid reliability mode: ' + str(spec['reliability']))

    config.IDENTITY_PRESERVATION = CN()
    config.IDENTITY_PRESERVATION.LAMBDA_RAW = float(spec['lambda_raw'])
    config.IDENTITY_PRESERVATION.LAMBDA_PRES = float(spec['lambda_pres'])
    config.IDENTITY_PRESERVATION.LAMBDA_EXCL = float(spec['lambda_excl'])
    config.IDENTITY_PRESERVATION.LAMBDA_RANK = float(spec['lambda_rank'])
    config.IDENTITY_PRESERVATION.MARGIN = float(spec['margin'])
    config.IDENTITY_PRESERVATION.EOT_ADDITIVE_GUIDANCE = False
    config.IDENTITY_PRESERVATION.SEMANTIC_EPSILON = 1e-6

    config.SEED = int(seed)
    config.TRAIN.AMP = True
    config.OUTPUT = str(output)
    config.TAG = 'pdf-identity-preservation-v2'
    config.freeze()
    return config
