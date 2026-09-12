# Identity-preservation tensor audit

Audit performed before the V2 implementation changes on frozen source commit
`282e4ca50ce0453d55ec39d7652f60ffbbbe5600` in the new worktree
`pdf-identity-preservation-v2`.

## Runtime confirmation

A real PRCC training batch was passed through the active ViT-B/16 training
forward at 384x128, batch size 64, world size 1, with `semantic_guidance=None`
so the old EOT additive path was not active during this probe.

| Tensor | Actual variable / expression | Shape | Finite |
|---|---|---:|---:|
| projected visual CLS | `img_feature_proj = x[:, 0, :]` | `[64, 512]` | yes |
| identity feature used by PDF | `feat_proj = self.bottleneck_proj(img_feature_proj)` | `[64, 512]` | yes |
| removed/text-guided component | `com_proj = self.bottleneck_proj(EOT(self.combine(x, t)))` | `[64, 512]` | yes |
| residual 1 | `ir_features = feat_proj - alpha * com_proj` | `[64, 512]` | yes |
| residual 2 | `ir_features_pos = feat_proj - alpha_pos * com_proj` | `[64, 512]` | yes |

The runtime input and token shapes were `[64, 3, 384, 128]` and `[64, 77]`.
The ViT projected token sequence is `[64, 193, 512]` before selecting its
CLS token: 24x8 patches plus one CLS token.

## Source-level data flow

- `models/clip_model.py:472-493`: the ViT returns `x` after the learned
  visual projection (`[..., 768] @ proj -> [..., 512]`) and `z` before that
  projection.
- `models/clip_model.py:1198-1202`: the active ViT branch sets
  `img_feature_proj = x[:, 0, :]`.
- `models/clip_model.py:773-775`: `bottleneck_proj` is a shared
  `BatchNorm1d(512)`; its bias is frozen, while its affine weight remains
  trainable.
- `models/clip_model.py:1206`: `feat_proj` is the BN output of the projected
  visual CLS feature. Therefore the PDF feature used by the current training
  losses is post-BN, not the pre-BN `img_feature_proj`.
- `models/clip_model.py:1218-1228`: the original caption is encoded by
  `encode_text_irra`, fused with visual tokens by `combine`, selected at the
  caption EOT position, and passed through the same `bottleneck_proj` to form
  `com_proj`.
- `train.py:101-114`: two independent per-sample coefficients are sampled as
  `alpha = randn(B,1)*0.5+0.5` and
  `alpha_pos = randn(B,1)*0.5+0.5`; the active residual names are
  `ir_features` and `ir_features_pos`.
- `train.py:112-115`: subtraction occurs after both `feat_proj` and `com_proj`
  have passed through BatchNorm, exactly as
  `f_1=f-a_1c` and `f_2=f-a_2c` with `f=feat_proj` and `c=com_proj`.

## V2 implementation consequence

The new parameter-free semantic losses must use the post-BN tensors returned
by the actual training forward:

```text
f       = feat_proj
c       = com_proj
f_1     = ir_features
f_2     = ir_features_pos
```

No new projection, attention, encoder, or trainable parameter is needed.
The frozen attribute semantic vector is applied only in the new identity
losses; it must not be passed as `semantic_guidance` to `clip_model`, and the
original caption clothing branch remains unchanged.
