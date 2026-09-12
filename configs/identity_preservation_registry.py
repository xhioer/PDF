"""Pre-registered V2 screening and matched multi-seed configurations.

This module is intentionally the only source of formal V2 hyperparameters.
Do not infer or add configurations from intermediate test results.
"""

from collections import OrderedDict


def _spec(mechanism, reliability='none', lambda_raw=0.0, lambda_pres=0.0,
          lambda_excl=0.0, lambda_rank=0.0, margin=0.10):
    return dict(mechanism=mechanism, reliability=reliability,
                lambda_raw=float(lambda_raw), lambda_pres=float(lambda_pres),
                lambda_excl=float(lambda_excl), lambda_rank=float(lambda_rank),
                margin=float(margin))


PRE_REGISTERED = OrderedDict([
    ('N00', _spec('Original PDF / control')),
    ('N01', _spec('Raw Visual Alignment', lambda_raw=0.05)),
    ('N02', _spec('Residual Identity Preservation', lambda_pres=0.02)),
    ('N03', _spec('Residual Identity Preservation', lambda_pres=0.05)),
    ('N04', _spec('Residual Identity Preservation', lambda_pres=0.10)),
    ('N05', _spec('Identity Leakage Exclusion', lambda_excl=0.02)),
    ('N06', _spec('Identity Leakage Exclusion', lambda_excl=0.05)),
    ('N07', _spec('Identity Leakage Exclusion', lambda_excl=0.10)),
    ('N08', _spec('Relative Semantic Ranking', lambda_rank=0.02, margin=0.10)),
    ('N09', _spec('Relative Semantic Ranking', lambda_rank=0.05, margin=0.10)),
    ('N10', _spec('Relative Semantic Ranking', lambda_rank=0.05, margin=0.20)),
    ('N11', _spec('Relative Semantic Ranking', lambda_rank=0.10, margin=0.20)),
    ('N12', _spec('Preservation + Exclusion', lambda_pres=0.05, lambda_excl=0.02)),
    ('N13', _spec('Preservation + Exclusion', lambda_pres=0.05, lambda_excl=0.05)),
    ('N14', _spec('Relative Semantic Ranking + Attribute Reliability',
                 reliability='attribute', lambda_rank=0.05, margin=0.10)),
    ('N15', _spec('Relative Semantic Ranking + Confidence Reliability',
                 reliability='confidence', lambda_rank=0.05, margin=0.10)),
    ('N16', _spec('Relative Semantic Ranking + Joint Reliability',
                 reliability='joint', lambda_rank=0.05, margin=0.10)),
])

SEED0_VARIANTS = tuple(PRE_REGISTERED)
MULTISEED_VARIANTS = ('N00', 'N09', 'N14', 'N15', 'N16')
SEEDS = (0, 1, 2)

OUTPUT_NAMES = {
    'N00': 'N00_control',
    'N01': 'N01_raw_align',
    'N02': 'N02_pres_002',
    'N03': 'N03_pres_005',
    'N04': 'N04_pres_010',
    'N05': 'N05_excl_002',
    'N06': 'N06_excl_005',
    'N07': 'N07_excl_010',
    'N08': 'N08_rank_002_m01',
    'N09': 'N09_rank_005_m01',
    'N10': 'N10_rank_005_m02',
    'N11': 'N11_rank_010_m02',
    'N12': 'N12_pres005_excl002',
    'N13': 'N13_pres005_excl005',
    'N14': 'N14_rank_attribute',
    'N15': 'N15_rank_confidence',
    'N16': 'N16_rank_joint',
}


def spec_for(variant):
    try:
        return dict(PRE_REGISTERED[variant])
    except KeyError:
        raise ValueError('Unknown pre-registered V2 variant: ' + str(variant))


def output_name(variant, seed):
    if variant not in PRE_REGISTERED:
        raise ValueError('Unknown pre-registered V2 variant: ' + str(variant))
    return OUTPUT_NAMES[variant] if int(seed) == 0 else variant


def queue_tasks():
    """Return fixed priority order: all seed0, then matched seed1, seed2."""
    return ([(variant, 0) for variant in SEED0_VARIANTS]
            + [(variant, 1) for variant in MULTISEED_VARIANTS]
            + [(variant, 2) for variant in MULTISEED_VARIANTS])
