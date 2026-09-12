"""No optimizer updates: isolate initial AMP overflow on the same V0/V4 batch.

Scale=1 is a diagnostic backward probe, NOT a changed training configuration.
The failed smoke experiment stays stopped and is never resumed here.
"""
import json
import runpy
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import distributed as dist
from data import build_dataloader
from data.semantic_reliability import aggregate_identity
from models.clip_model import build_CLIP_from_openai_pretrained
from models.utils.simple_tokenizer import SimpleTokenizer
from losses import build_losses
from losses.orthogonal_loss import OrthogonalProjectionLoss
from tools.run_reliability import build_bank
from tools.utils import set_seed
from train import tokenize


def main():
    cfg = runpy.run_path(str(ROOT / 'configs/v4_p2_joint.py'))['CONFIG']
    with tempfile.TemporaryDirectory(prefix='pdf-amp-diagnostic-') as directory:
        torch.cuda.set_device(0)
        dist.init_process_group('nccl', init_method='file://' + directory + '/store', world_size=1, rank=0)
        set_seed(cfg.SEED)
        loader, *rest = build_dataloader(cfg)
        model, _ = build_CLIP_from_openai_pretrained('ViT-B/16', (384, 128), 16, 150)
        model.eval().float()
        model.freeze_text_encoder()
        model.cuda()
        bank = build_bank(model, loader.dataset)
        cla, pair, _, _ = build_losses(cfg, 300)
        opl = OrthogonalProjectionLoss()
        imgs, pids, camids, clothes, payload = next(iter(loader))
        imgs, pids, clothes = imgs.cuda(), pids.cuda(), clothes.cuda()
        tokens = tokenize(payload['caption'], SimpleTokenizer(), truncate=True).cuda()
        guidance = aggregate_identity(bank[payload['semantic_indices'].cuda()], payload['reliability'].cuda())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        bn_state = {key: value.clone() for key, value in model.bottleneck_proj.state_dict().items()}
        results = []
        for variant, sem in [('V0', None), ('V4', guidance)]:
            for scale in (65536.0, 1.0):
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                model.bottleneck_proj.load_state_dict(bn_state)
                model.zero_grad(set_to_none=True)
                model.train()
                with torch.cuda.amp.autocast():
                    scores, features, combined = model(imgs, tokens, semantic_guidance=sem)
                    f, c = features[1], combined[0]
                    id_loss = cla(scores[1], pids)
                    a = torch.randn(f.size(0), 1, device=f.device, dtype=f.dtype) * 0.5 + 0.5
                    b = torch.randn(f.size(0), 1, device=f.device, dtype=f.dtype) * 0.5 + 0.5
                    pair_loss = pair(f - c * a, f - c * b, pids)
                    opl_loss = opl(f, c, pids, clothes) * 0.5
                    loss = id_loss + pair_loss + opl_loss
                (loss * scale).backward()
                bad = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        gradient = parameter.grad
                        n_nan = torch.isnan(gradient).sum().item()
                        n_inf = torch.isinf(gradient).sum().item()
                        if n_nan or n_inf:
                            bad.append(dict(parameter=name, nan_count=n_nan, inf_count=n_inf))
                row = dict(variant=variant, backward_scale=scale,
                    optimizer_steps=0, loss=loss.item(), id_loss=id_loss.item(), pair_loss=pair_loss.item(),
                    pdf_opl_loss=opl_loss.item(), loss_finite=bool(torch.isfinite(loss)),
                    all_gradients_finite=not bad, nonfinite_parameter_count=len(bad),
                    nonfinite_parameters=bad)
                results.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != 'nonfinite_parameters'}), flush=True)
        report = dict(purpose='diagnostic only; zero optimizer steps; failed smoke not resumed',
            config='configs/v4_p2_joint.py', batch_size=64, seed=0, same_images=True,
            reset_rng_and_bn_before_each_probe=True,
            original_amp_default_initial_scale=65536.0,
            scale_one_is_not_a_training_protocol_change=True, results=results)
        (ROOT / 'reports/reliability_amp_diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
