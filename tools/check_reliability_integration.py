"""Exercise actual pretrained CLIP fusion, zero-evidence fallback and inference."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from data.semantic_reliability import ATTRIBUTES, MODES, aggregate_identity, reliability, attribute_phrase
from models.clip_model import build_CLIP_from_openai_pretrained
from models.utils.simple_tokenizer import SimpleTokenizer
from train import tokenize


def main():
    torch.manual_seed(0)
    model, _ = build_CLIP_from_openai_pretrained('ViT-B/16', (384, 128), 16, 150)
    model.float().cuda().freeze_text_encoder()
    keys = tuple(model.state_dict())
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = SimpleTokenizer()
    descriptions = [dict(zip(ATTRIBUTES, ('male', 'black', 'short', 'average'))),
                    dict(zip(ATTRIBUTES, ('female', 'unknown', 'long', 'slim')))]
    for row in descriptions:
        row.update({key + '_confidence': level for key, level in zip(ATTRIBUTES, ('high', 'medium', 'low', 'medium'))})
    embeddings = torch.zeros(2, 4, 512, device='cuda')
    with torch.no_grad():
        for i, row in enumerate(descriptions):
            for j, key in enumerate(ATTRIBUTES):
                if row[key] != 'unknown':
                    text = tokenize([attribute_phrase(key, row[key])], tokenizer).cuda()
                    embeddings[i, j] = model.encode_text(text)[0]
        guides = {mode: aggregate_identity(embeddings, torch.tensor([
            reliability(row, mode) for row in descriptions], device='cuda')) for mode in MODES}
        zero = aggregate_identity(embeddings, torch.zeros(2, 4, device='cuda'))
        assert torch.count_nonzero(zero).item() == 0
        images = torch.randn(2, 3, 384, 128, device='cuda')
        captions = tokenize(['A man wearing a white shirt.', 'A woman wearing a black coat.'], tokenizer).cuda()
        model.eval()
        original = model(images)
        guided_eval = model(images, semantic_guidance=guides['joint'])
        torch.testing.assert_close(original, guided_eval, rtol=0, atol=0)
        assert original.shape == (2, 512)
        model.train()
        model.bottleneck_proj.eval()  # isolate fusion without altering BN running statistics
        baseline = model(images, captions)
        absent = model(images, captions, semantic_guidance=zero)
        torch.testing.assert_close(baseline[2][0], absent[2][0], rtol=0, atol=0)
        fused = {mode: model(images, captions, semantic_guidance=value)[2][0] for mode, value in guides.items()}
        differences = {a + '_vs_' + b: (fused[a] - fused[b]).abs().max().item()
                       for i, a in enumerate(MODES) for b in MODES[i + 1:]}
        assert all(value > 1e-6 for value in differences.values()), differences
        assert keys == tuple(model.state_dict())
        assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = dict(passed=True, added_trainable_parameters=0, trainable_parameters=count,
        inference_shape=[2, 512], inference_bitwise_equal=True,
        all_zero_evidence_equals_original_guidance=True,
        unknown_embedding_not_encoded=True, mode_fusion_max_absolute_differences=differences)
    (ROOT / 'reports/reliability_integration_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
