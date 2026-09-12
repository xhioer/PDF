"""Parameter-free identity-semantic preservation losses for PDF V2."""

from typing import Dict

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, valid: torch.Tensor,
                 reference: torch.Tensor) -> torch.Tensor:
    """Average only valid semantic samples, retaining a zero-gradient path."""
    if bool(valid.any().item()):
        return values[valid].mean()
    return reference.float().sum() * 0.0


def identity_preservation_losses(
    feat_proj: torch.Tensor,
    com_proj: torch.Tensor,
    residual_1: torch.Tensor,
    residual_2: torch.Tensor,
    semantic_vector: torch.Tensor,
    semantic_valid: torch.Tensor,
    margin: float,
) -> Dict[str, torch.Tensor]:
    """Return the four requested losses and diagnostics.

    All cosine inputs are explicitly cast to float32 and L2-normalized. The
    caller supplies a boolean validity mask; invalid/all-unknown samples do
    not enter any semantic mean or denominator.
    """
    if semantic_vector.ndim != 2:
        raise ValueError('semantic_vector must have shape [B,D]')
    if semantic_valid.ndim != 1 or semantic_valid.shape[0] != semantic_vector.shape[0]:
        raise ValueError('semantic_valid must have shape [B]')
    for name, value in (('feat_proj', feat_proj), ('com_proj', com_proj),
                        ('residual_1', residual_1), ('residual_2', residual_2)):
        if value.shape != semantic_vector.shape:
            raise ValueError('{} shape {} does not match semantic vector {}'.format(
                name, tuple(value.shape), tuple(semantic_vector.shape)))

    valid = semantic_valid.to(device=semantic_vector.device, dtype=torch.bool)
    f = F.normalize(feat_proj.float(), p=2, dim=-1, eps=1e-12)
    c = F.normalize(com_proj.float(), p=2, dim=-1, eps=1e-12)
    r1 = F.normalize(residual_1.float(), p=2, dim=-1, eps=1e-12)
    r2 = F.normalize(residual_2.float(), p=2, dim=-1, eps=1e-12)
    s = F.normalize(semantic_vector.float(), p=2, dim=-1, eps=1e-12)

    cos_raw = (f * s).sum(dim=-1)
    cos_com = (c * s).sum(dim=-1)
    cos_res1 = (r1 * s).sum(dim=-1)
    cos_res2 = (r2 * s).sum(dim=-1)
    margin_res1 = cos_res1 - cos_com
    margin_res2 = cos_res2 - cos_com

    raw_values = 1.0 - cos_raw
    pres_values = 0.5 * ((1.0 - cos_res1) + (1.0 - cos_res2))
    excl_values = cos_com.square()
    rank_values = 0.5 * (
        F.relu(float(margin) + cos_com - cos_res1)
        + F.relu(float(margin) + cos_com - cos_res2)
    )

    zero = semantic_vector.float().sum() * 0.0
    return {
        'L_raw': _masked_mean(raw_values, valid, semantic_vector),
        'L_pres': _masked_mean(pres_values, valid, semantic_vector),
        'L_excl': _masked_mean(excl_values, valid, semantic_vector),
        'L_rank': _masked_mean(rank_values, valid, semantic_vector),
        'cos_raw_sem': _masked_mean(cos_raw, valid, semantic_vector),
        'cos_res1_sem': _masked_mean(cos_res1, valid, semantic_vector),
        'cos_res2_sem': _masked_mean(cos_res2, valid, semantic_vector),
        'cos_com_sem': _masked_mean(cos_com, valid, semantic_vector),
        'semantic_margin_res1_minus_com': _masked_mean(margin_res1, valid, semantic_vector),
        'semantic_margin_res2_minus_com': _masked_mean(margin_res2, valid, semantic_vector),
        'semantic_valid_count': valid.sum().to(dtype=torch.float32),
        'semantic_valid_rate': valid.to(dtype=torch.float32).mean() if valid.numel() else zero,
    }


def weighted_identity_semantic_loss(terms: Dict[str, torch.Tensor],
                                    lambda_raw: float,
                                    lambda_pres: float,
                                    lambda_excl: float,
                                    lambda_rank: float) -> torch.Tensor:
    """Combine only the pre-registered scalar weights."""
    return (float(lambda_raw) * terms['L_raw']
            + float(lambda_pres) * terms['L_pres']
            + float(lambda_excl) * terms['L_excl']
            + float(lambda_rank) * terms['L_rank'])
