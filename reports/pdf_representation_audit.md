# PDF representation audit
Generated from the checked-out `models/clip_model.py` and the validated R00 checkpoint state dictionaries; no paper-level representation definition was substituted.
## Source and architecture
- Code file: `/data/projects/PDF-worktrees/pdf-representation-diagnosis/models/clip_model.py`
- Code SHA256: `9b3a3489c929518677ce10ae9101c8ba0e31bc6c31f36a1348ffc34107ef5aa7`
- Diagnosis worktree branch: `pdf-representation-diagnosis`
- Checkpoint-derived visual state keys: `323`
- Visual backbone: `VisionTransformer` (the checkpoint has `visual.conv1.weight` with shape `[768, 3, 16, 16]`).
- Input resolution: `(384, 128)`; patch/stride: `(16,16)`; grid: `24 x 8`; tokens: `192 patches + CLS = 193`.
- Visual transformer: `12` blocks, width `768`, heads `12` (`width // 64`); visual projection `[768, 512]` maps width `768` to dimension `512`.
- Text transformer used by `encode_text_irra`: `12` blocks, width `512`, heads `8` (`width // 64`). This head count is taken from the current `CLIP` constructor, not inferred from the in-projection matrix shape.
- Final classification/evaluation feature dimension: `512`; `bottleneck_proj` is `BatchNorm1d(512)`; classifier is `512 -> 150`.

## Actual forward variables
The ViT forward returns `(x, z)`. `z` is the post-transformer, post-`ln_post` token sequence in width 768. `x` is `z @ visual.proj`, in dimension 512. The diagnosis defines:

- `H_0 ... H_12`: the CLS token before the transformer and after each actual visual transformer block, captured with hooks in the current code.
- `Z_l = visual.proj(ln_post(H_l))`, dimension 512. `Z_12` is exactly the current final projected visual CLS before BN; `Z_0 ... Z_11` are the same code-defined output-space map applied offline to the corresponding hooked CLS state so layers are comparable.
- `F_visual = bottleneck_proj(Z_12)`, eval-mode BN output, dimension 512. This is the image-only `model(image)` output in the actual eval branch.
- `F_cloth = com_proj = bottleneck_proj(EOT(combine(x, encode_text_irra(caption))))`, dimension 512. This is the training-only cloth-related path, evaluated deterministically with the checkpoint BN running statistics.
- No `F_ir` is guessed or sampled during extraction. It is constructed offline only as `normalize(F_visual - alpha * F_cloth)` in the exact BN-output subtraction space.

## Actual subtraction and alpha generation
The frozen PDF training code uses `ir_features = reference_features_proj - com_proj * alpha`, where both operands are 512-dimensional BN outputs. `alpha` is sampled per sample as `torch.randn(B, 1) * 0.5 + 0.5`; the positive branch draws an independent alpha. This diagnosis does not alter that behavior and does not emulate an unrecorded random draw. It uses the pre-registered offline sweep `alpha ∈ [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`.

## Projection and inference audit
- Visual CLS receives `visual.proj` before `bottleneck_proj`; cloth `combine` is already dimension 512 and then receives the same `bottleneck_proj`.
- Official image-only inference in the eval branch returns the BN output `F_visual`; the text-conditioned eval branch returns `F_visual + BN(combine(...))`, but it is not used as the image representation here.
- `encode_image()` is not used because this checkout's helper predates the local `(projected_tokens, hidden_tokens)` visual return signature. The extraction calls `model.visual` directly and reproduces the forward variables above.
- All diagnostic features are saved raw and L2-normalized; no training mode, optimizer, loss, classifier, prompt, relation branch, or new model head is invoked.

## Data and scope
- Dataset scope: PRCC TRAIN only, using the existing 17,896-image TRAIN inventory. No PRCC TEST metric is read.
- H1 probe/alignment split: the frozen 148-ID / 888-image manifest, 100 identity-disjoint semantic-dev IDs (600 images) and 48 semantic-val IDs (288 images).
- The existing parsing cache contains only PRCC TEST masks in this environment; no TRAIN parsing cache was found, so region analysis is skipped.
