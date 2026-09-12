"""Real-batch gradient and invariance checks for PDF identity-preservation V2."""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from configs.identity_preservation_common import build_config
from configs.identity_preservation_registry import spec_for
from data import build_dataloader
from data.semantic_reliability import aggregate_identity
from losses.identity_preservation import identity_preservation_losses
from models.clip_model import build_CLIP_from_openai_pretrained
from models.utils.simple_tokenizer import SimpleTokenizer
from tools.run_identity_preservation import build_bank
from tools.utils import set_seed
from train import tokenize


def main():
    output_path = ROOT / 'reports/identity_preservation_gradient_sanity.json'
    spec = spec_for('N16')
    result = dict(
        passed=False,
        commit=os.popen('git rev-parse HEAD').read().strip(),
        batch_search_limit=20,
        loss_names=['L_raw', 'L_pres', 'L_excl', 'L_rank'],
        eot_additive_guidance=False,
        no_new_trainable_projection=True,
    )
    loader = None
    with tempfile.TemporaryDirectory(prefix='pdf-identity-gradient-sanity-') as directory:
        dist.init_process_group('nccl', init_method='file://' + directory + '/rendezvous',
                                rank=0, world_size=1)
        try:
            torch.cuda.set_device(0)
            config = build_config(spec, 0, ROOT / 'outputs/pdf_identity_preservation_v2/sanity')
            config.defrost()
            config.DATA.NUM_WORKERS = 0
            config.freeze()
            set_seed(config.SEED)
            loader, *_ = build_dataloader(config)
            dataset = loader.dataset
            batch = None
            for batch_index, candidate in enumerate(loader):
                payload = candidate[-1]
                valid = payload['reliability'].sum(dim=1) > 0.0
                if bool(valid.any().item()):
                    batch = (batch_index, candidate)
                    break
                if batch_index + 1 >= result['batch_search_limit']:
                    break
            if batch is None:
                raise AssertionError('No semantic-valid real batch found')
            batch_index, (images, pids, camids, clothes_ids, payload) = batch
            result['batch_index'] = batch_index
            result['batch_size'] = int(images.shape[0])
            result['input_shape'] = list(images.shape)

            model, _ = build_CLIP_from_openai_pretrained(
                'ViT-B/16', (config.DATA.HEIGHT, config.DATA.WIDTH), 16, 150)
            model.float().cuda()
            model.freeze_text_encoder()
            trainable_names_before = [name for name, p in model.named_parameters() if p.requires_grad]
            trainable_count_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
            bank = build_bank(model, dataset)
            indices = payload['semantic_indices'].cuda(non_blocking=True)
            weights = payload['reliability'].cuda(non_blocking=True)
            semantic_vector = aggregate_identity(bank[indices], weights)
            semantic_valid = weights.sum(dim=1) > 0.0
            result['semantic_valid_count'] = int(semantic_valid.sum().item())
            result['semantic_valid_rate'] = float(semantic_valid.float().mean().item())

            tokenizer = SimpleTokenizer()
            text = tokenize(payload['caption'], tokenizer, context_length=77, truncate=True).cuda()
            model.eval()
            with torch.no_grad():
                image_only = model(images.cuda())
                image_only_with_semantic_argument = model(
                    images.cuda(), semantic_guidance=torch.zeros(images.shape[0], 512, device='cuda'))
            result['image_only_shape'] = list(image_only.shape)
            result['image_only_bitwise_equal_with_semantic_argument'] = bool(
                torch.equal(image_only, image_only_with_semantic_argument))
            if not result['image_only_bitwise_equal_with_semantic_argument']:
                raise AssertionError('Image-only inference changed when semantic argument was supplied')

            model.train()
            with torch.cuda.amp.autocast():
                _, features, components = model(images.cuda(), text)
                _, feat_proj = features
                com_proj = components[0]
                alpha = torch.randn(feat_proj.size(0), 1, device=feat_proj.device,
                                    dtype=feat_proj.dtype) * 0.5 + 0.5
                alpha_pos = torch.randn(feat_proj.size(0), 1, device=feat_proj.device,
                                        dtype=feat_proj.dtype) * 0.5 + 0.5
                residual_1 = feat_proj - com_proj * alpha
                residual_2 = feat_proj - com_proj * alpha_pos
                terms = identity_preservation_losses(
                    feat_proj, com_proj, residual_1, residual_2,
                    semantic_vector, semantic_valid, margin=0.10)

            required_finite = ['L_raw', 'L_pres', 'L_excl', 'L_rank']
            result['loss_finite'] = {
                name: bool(torch.isfinite(terms[name]).item()) for name in required_finite
            }
            if not all(result['loss_finite'].values()):
                raise FloatingPointError('Non-finite identity-preservation loss')

            gradient_rows = {}
            visual_names = [name for name, _ in model.named_parameters()
                            if name.startswith('visual.') and name in trainable_names_before]
            for name in required_finite:
                model.zero_grad(set_to_none=True)
                terms[name].backward(retain_graph=True)
                visual_gradients = []
                all_gradients = []
                for parameter_name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    finite = bool(torch.isfinite(parameter.grad).all().item())
                    norm = float(parameter.grad.detach().float().norm().item())
                    all_gradients.append((parameter_name, finite, norm))
                    if parameter_name in visual_names:
                        visual_gradients.append((parameter_name, finite, norm))
                nonzero_visual = [row for row in visual_gradients if row[2] > 0.0]
                if not nonzero_visual or not all(row[1] for row in all_gradients):
                    raise AssertionError('Missing/non-finite visual gradient for ' + name)
                gradient_rows[name] = dict(
                    visual_trainable_parameter_tensors=len(visual_names),
                    visual_gradient_tensors=len(visual_gradients),
                    visual_nonzero_gradient_tensors=len(nonzero_visual),
                    all_nonzero_gradient_norm_max=max(row[2] for row in all_gradients),
                    visual_nonzero_gradient_norm_max=max(row[2] for row in nonzero_visual),
                    all_gradient_finite=True,
                )
            result['gradient_checks'] = gradient_rows

            frozen_prefixes = ('token_embedding.', 'transformer.', 'ln_final.')
            frozen_parameter_checks = {}
            for name, parameter in model.named_parameters():
                if name == 'positional_embedding' or name == 'text_projection' \
                        or name.startswith(frozen_prefixes):
                    frozen_parameter_checks[name] = not parameter.requires_grad
            result['frozen_text_encoder'] = frozen_parameter_checks
            if not all(frozen_parameter_checks.values()):
                raise AssertionError('A text encoder parameter became trainable')
            result['trainable_parameter_names_unchanged'] = (
                trainable_names_before == [name for name, p in model.named_parameters() if p.requires_grad])
            result['trainable_parameter_count_before'] = trainable_count_before
            result['trainable_parameter_count_after'] = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            if not result['trainable_parameter_names_unchanged']:
                raise AssertionError('Trainable parameter set changed')
            result['passed'] = True
        finally:
            if loader is not None:
                loader.shutdown()
            if dist.is_initialized():
                dist.destroy_process_group()
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
