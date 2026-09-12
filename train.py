import copy
import os
import time
import datetime
import logging
import torch
from torch import optim, nn
from tools.utils import AverageMeter
import numpy as np
import torch.nn.functional as F
from torch import distributed as dist
from tqdm import tqdm
from models.utils.simple_tokenizer import SimpleTokenizer
from typing import Any, Union, List
from pkg_resources import packaging
from losses.gather import GatherLayer
from losses.orthogonal_loss import OrthogonalProjectionLoss
from losses.triplet_loss import TripletLoss
from losses.identity_preservation import (
    identity_preservation_losses,
    weighted_identity_semantic_loss,
)


def tokenize(texts: Union[str, List[str]], tokenizer, context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def train_clip_combiner(config, epoch, clip_model, criterion_cla, criterion_pair, optimizer, trainloader, pid2clothes,
                        semantic_bank=None, observer=None, stop_at=None, allow_amp_overflow=False,
                        identity_config=None, epoch_observer=None):
    logger = logging.getLogger('cir_reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_opl_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    crossentropy_criterion = nn.CrossEntropyLoss()
    tokenizer = SimpleTokenizer()
    opl = OrthogonalProjectionLoss()
    triplet_criterion = TripletLoss(margin=0.3)
    scaler = torch.cuda.amp.GradScaler()
    identity_config = identity_config or {}
    def config_value(name, default):
        if hasattr(identity_config, name.upper()):
            return float(getattr(identity_config, name.upper()))
        return float(identity_config.get(name, default))
    lambda_raw = config_value('lambda_raw', 0.0)
    lambda_pres = config_value('lambda_pres', 0.0)
    lambda_excl = config_value('lambda_excl', 0.0)
    lambda_rank = config_value('lambda_rank', 0.0)
    margin = config_value('margin', 0.10)
    if min(lambda_raw, lambda_pres, lambda_excl, lambda_rank) < 0:
        raise ValueError('Identity-preservation lambdas must be non-negative')
    diagnostic_keys = ('L_raw', 'L_pres', 'L_excl', 'L_rank',
                       'cos_raw_sem', 'cos_res1_sem', 'cos_res2_sem', 'cos_com_sem',
                       'semantic_margin_res1_minus_com', 'semantic_margin_res2_minus_com')
    epoch_sums = {key: 0.0 for key in diagnostic_keys}
    epoch_valid_count = 0
    epoch_sample_count = 0
    clip_model.train()
    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, cap) in enumerate(trainloader):
        semantic_vector = None
        semantic_valid = None
        if isinstance(cap, dict):
            if semantic_bank is None:
                if lambda_raw or lambda_pres or lambda_excl or lambda_rank:
                    raise ValueError('Identity-preservation loss requires a frozen semantic embedding bank')
            else:
                from data.semantic_reliability import aggregate_identity
                device = torch.device('cuda')
                semantic_indices = cap['semantic_indices'].to(device=device, non_blocking=True)
                reliability_weights = cap['reliability'].to(device=device, non_blocking=True)
                semantic_vector = aggregate_identity(semantic_bank[semantic_indices], reliability_weights)
                semantic_valid = reliability_weights.sum(dim=1) > 0.0
            cap = cap['caption']
        images_in_batch = imgs.size(0)
        optimizer.zero_grad()
        reference_images = imgs.cuda()
        imgs, pids, clothes_ids = imgs.cuda(), pids.cuda(), clothes_ids.cuda()
        text_inputs = tokenize(cap, tokenizer, context_length=77, truncate=True).cuda()
        with torch.cuda.amp.autocast():
            [cls_score, cls_score_proj], [img_feature, img_feature_proj], [com_proj, com_z, t_bn, t_z_bn] = clip_model(
                reference_images, text_inputs)
            reference_features_proj = img_feature_proj
            cla_loss = criterion_cla(cls_score_proj, pids)
            # Use two independent per-sample coefficients as in Eq. (10)-(11).
            alpha = torch.randn(
                reference_features_proj.size(0), 1,
                device=reference_features_proj.device,
                dtype=reference_features_proj.dtype,
            ) * 0.5 + 0.5
            alpha_pos = torch.randn(
                reference_features_proj.size(0), 1,
                device=reference_features_proj.device,
                dtype=reference_features_proj.dtype,
            ) * 0.5 + 0.5
            ir_features = reference_features_proj - com_proj * alpha
            ir_features_pos = reference_features_proj - com_proj * alpha_pos
            cir_com_loss = criterion_pair(ir_features, ir_features_pos, pids)
            opl_loss = opl(reference_features_proj, com_proj, pids, clothes_ids)*0.5
            if semantic_vector is not None:
                semantic_terms = identity_preservation_losses(
                    reference_features_proj, com_proj, ir_features, ir_features_pos,
                    semantic_vector, semantic_valid, margin)
                semantic_loss = weighted_identity_semantic_loss(
                    semantic_terms, lambda_raw, lambda_pres, lambda_excl, lambda_rank)
            else:
                semantic_terms = None
                semantic_loss = reference_features_proj.float().sum() * 0.0
            loss = cla_loss + cir_com_loss + opl_loss + semantic_loss

        epoch_sample_count += images_in_batch
        if semantic_terms is not None:
            valid_count = int(semantic_terms['semantic_valid_count'].detach().item())
            epoch_valid_count += valid_count
            for key in diagnostic_keys:
                epoch_sums[key] += float(semantic_terms[key].detach().item()) * valid_count

        # Backpropagate and update the weights
        if observer is not None and not torch.isfinite(torch.stack((loss, cla_loss, cir_com_loss, opl_loss))).all():
            raise FloatingPointError('Non-finite loss at epoch {} batch {}'.format(epoch + 1, batch_idx + 1))
        scale_before = scaler.get_scale() if observer is not None else None
        scaler.scale(loss).backward()
        if observer is not None:
            scaler.unscale_(optimizer)
            gradient_finite = not any(value.item() != 0 for value in scaler._found_inf_per_device(optimizer).values())
            if not gradient_finite and not allow_amp_overflow:
                raise FloatingPointError('Non-finite gradient at epoch {} batch {}'.format(epoch + 1, batch_idx + 1))
        scaler.step(optimizer)
        scaler.update()

        if observer is not None:
            torch.cuda.synchronize()
            observer(dict(epoch=epoch + 1, iteration=batch_idx + 1,
                          train_loss=loss.item(), id_loss=cla_loss.item(),
                          triplet_loss=0.0, pair_loss=cir_com_loss.item(),
                          pdf_opl_loss=opl_loss.item(), semantic_loss=semantic_loss.item(),
                          L_raw=semantic_terms['L_raw'].item() if semantic_terms else 0.0,
                          L_pres=semantic_terms['L_pres'].item() if semantic_terms else 0.0,
                          L_excl=semantic_terms['L_excl'].item() if semantic_terms else 0.0,
                          L_rank=semantic_terms['L_rank'].item() if semantic_terms else 0.0,
                          cos_raw_sem=semantic_terms['cos_raw_sem'].item() if semantic_terms else 0.0,
                          cos_res1_sem=semantic_terms['cos_res1_sem'].item() if semantic_terms else 0.0,
                          cos_res2_sem=semantic_terms['cos_res2_sem'].item() if semantic_terms else 0.0,
                          cos_com_sem=semantic_terms['cos_com_sem'].item() if semantic_terms else 0.0,
                          semantic_margin_res1_minus_com=(semantic_terms['semantic_margin_res1_minus_com'].item()
                                                          if semantic_terms else 0.0),
                          semantic_margin_res2_minus_com=(semantic_terms['semantic_margin_res2_minus_com'].item()
                                                          if semantic_terms else 0.0),
                          semantic_valid_rate=(semantic_terms['semantic_valid_rate'].item()
                                               if semantic_terms else 0.0),
                          semantic_valid_count=(semantic_terms['semantic_valid_count'].item()
                                                if semantic_terms else 0.0),
                          learning_rate=optimizer.param_groups[0]['lr'],
                          scaler_scale_before=scale_before, scaler_scale_after=scaler.get_scale(),
                          optimizer_step_skipped=not gradient_finite,
                          unscaled_gradient_finite=gradient_finite,
                          memory_allocated=torch.cuda.memory_allocated(),
                          memory_reserved=torch.cuda.memory_reserved(),
                          iteration_seconds=time.time() - end,
                          batch_size=images_in_batch))

        batch_cla_loss.update(cla_loss.item(), reference_images.size(0))
        batch_pair_loss.update(cir_com_loss.item(), reference_images.size(0))
        batch_opl_loss.update(opl_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if stop_at is not None and time.monotonic() >= stop_at:
            break

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(trainloader):
            logger.info(
                'Epoch{0} [{1}/{2}] '
                'ClaLoss:{3:.4f} PairLoss:{4:.4f} OPLLoss:{5:.4f}'.format(
                    epoch + 1,
                    batch_idx + 1,
                    len(trainloader),
                    batch_cla_loss.avg,
                    batch_pair_loss.avg,
                    batch_opl_loss.avg,
                )
            )

    if epoch_observer is not None:
        denominator = float(epoch_valid_count) if epoch_valid_count else 1.0
        epoch_summary = dict(
            epoch=epoch + 1,
            semantic_valid_count=epoch_valid_count,
            semantic_valid_rate=(float(epoch_valid_count) / float(epoch_sample_count)
                                 if epoch_sample_count else 0.0),
        )
        for key in diagnostic_keys:
            epoch_summary[key] = epoch_sums[key] / denominator if epoch_valid_count else 0.0
        epoch_observer(epoch_summary)

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'OPLLoss:{mlm_loss.avg:.4f}'.format(
            epoch + 1, batch_time=batch_time, data_time=data_time,
            cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
            mlm_loss=batch_opl_loss))

