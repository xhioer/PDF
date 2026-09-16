# PDF Representation Leakage & Over-Suppression Diagnosis

**Protocol:** PDF-RLOS-Diagnosis  \  **Scope:** PRCC TRAIN only  \  **Seeds:** R00 seed0/seed1/seed2, epoch50 final

## Executive conclusion

- **H1 — cloth-component predictability inside the identity stream:** **MODERATE SUPPORT**. Final-minus-early validation cosine by seed: `0.0364, 0.0322, 0.0287`; late-minus-early: `0.0158, 0.0148, 0.0120`.

- **H2 — cross-clothing identity-correlated information in `F_cloth`:** **STRONG SUPPORT**. Different-clothes verification AUC by seed: `0.9734, 0.9690, 0.9726`; A-gallery/C-query R1 by seed: `82.96%, 82.83%, 82.81%`.

- **Recommended next method:** **A. Separated Attention + Structure Preservation**.

- **Over-suppression signal:** `NOT SUPPORTED` under the preregistered rule. Best alpha by seed: `1.50, 1.25, 1.00`; best-minus-alpha=1 R1: `0.015, 0.030, 0.000` pp; mAP: `0.062, 0.044, 0.000` pp.


Interpretation is intentionally bounded: H1 refers to information predictive of **PDF's own PDF-defined cloth-related component**, not ground-truth clothing leakage. H2 refers to cross-clothing identity-correlated information; without TRAIN parsing evidence it is not a claim that body structure has been proven in `F_cloth`.

## 1. PDF code audit

See [pdf_representation_audit.md](pdf_representation_audit.md). The actual code is a 12-block ViT-B/16 with 512-D projected visual tokens. `F_visual` is the eval BN output of visual CLS; `F_cloth` is the training-only `com_proj` route using the frozen original caption, `encode_text_irra`, cross-modal transformer/attention, EOT selection, and the same BN. Offline residuals use the exact BN-space formula `normalize(F_visual - alpha * F_cloth)`.

## 2. Baseline checkpoint provenance

See [baseline_checkpoint_manifest.json](baseline_checkpoint_manifest.json). All three files passed `run_id=R00`, `epoch=50`, and the actual payload role `primary epoch50 final`; seed checks also passed. Any `best_test_*` fields are recorded as auxiliary provenance only; they were not selected.

| seed | epoch | checkpoint SHA256 | source commit | path |
| --- | --- | --- | --- | --- |
| 0 | 50 | cb728c47c9f5… | 7c890687ce51… | /data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R00_control/epoch50_final.pth |
| 1 | 50 | 0d23ee880236… | fe2eba095b5b… | /data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs/crossclothes_relation_confirm/seed1/R00/epoch50_final.pth |
| 2 | 50 | 083cd935d4f0… | fe2eba095b5b… | /data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs/crossclothes_relation_confirm/seed2/R00/epoch50_final.pth |

## 3. Train split and pair manifests

The full extraction inventory contains `17896` PRCC TRAIN images and `150` identities. H1 uses the copied frozen semantic split: 100 dev IDs / 600 selected images and 48 val IDs / 288 selected images, with no identity overlap. H2 and alpha-sweep pair/retrieval diagnostics use full PRCC TRAIN. Fixed pair seed is `20260916`; the negative manifest contains `100000` negatives and is shared across all representations and seeds.

Split details: [split_resolution.json](../outputs/pdf_representation_diagnosis/shared/split_manifest/split_resolution.json); pairs: [pair_manifest.json](../outputs/pdf_representation_diagnosis/shared/pair_manifest/pair_manifest.json); negatives: [negative_manifest.json](../outputs/pdf_representation_diagnosis/shared/negative_manifest/negative_manifest.json).

## 4. Extraction validation

- seed0: `17896` images, `Z_0...Z_12`, raw/L2 files; `Z_12` versus actual visual output max error `0.0000000`; `F_visual` versus eval image-only forward max error `0.0000000`.

- seed1: `17896` images, `Z_0...Z_12`, raw/L2 files; `Z_12` versus actual visual output max error `0.0000000`; `F_visual` versus eval image-only forward max error `0.0000000`.

- seed2: `17896` images, `Z_0...Z_12`, raw/L2 files; `Z_12` versus actual visual output max error `0.0000000`; `F_visual` versus eval image-only forward max error `0.0000000`.

No model parameters were updated and no PRCC TEST path was consumed. Region analysis was skipped because no reliable TRAIN parsing cache exists; see [region_analysis_status.json](region_analysis_status.json).

## 5. H1-A — layer-wise linear predictability

Ridge probes are fit only on the 100-ID dev images, with fixed `ridge_alpha=1.0` and no validation hyperparameter search. The reported H1 curve is identity-disjoint val.

| layer | seed0 cosine | seed1 cosine | seed2 cosine | mean | std | R2 | NMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Z_0 | 0.9494 | 0.9535 | 0.9565 | 0.9532 | 0.0029 | -0.0139 | 0.0924 |
| Z_1 | 0.9266 | 0.9335 | 0.9376 | 0.9326 | 0.0045 | -0.4863 | 0.1356 |
| Z_2 | 0.9251 | 0.9314 | 0.9357 | 0.9307 | 0.0044 | -0.5361 | 0.1401 |
| Z_3 | 0.9251 | 0.9318 | 0.9349 | 0.9306 | 0.0041 | -0.5379 | 0.1402 |
| Z_4 | 0.9177 | 0.9250 | 0.9300 | 0.9242 | 0.0050 | -0.7012 | 0.1552 |
| Z_5 | 0.9102 | 0.9191 | 0.9222 | 0.9171 | 0.0051 | -0.8815 | 0.1715 |
| Z_6 | 0.9076 | 0.9178 | 0.9201 | 0.9152 | 0.0055 | -0.9366 | 0.1766 |
| Z_7 | 0.9084 | 0.9168 | 0.9203 | 0.9152 | 0.0050 | -0.9444 | 0.1773 |
| Z_8 | 0.9086 | 0.9164 | 0.9213 | 0.9154 | 0.0052 | -0.9538 | 0.1782 |
| Z_9 | 0.9162 | 0.9241 | 0.9280 | 0.9228 | 0.0049 | -0.7763 | 0.1620 |
| Z_10 | 0.9331 | 0.9403 | 0.9407 | 0.9380 | 0.0035 | -0.4200 | 0.1294 |
| Z_11 | 0.9483 | 0.9537 | 0.9540 | 0.9520 | 0.0026 | -0.1021 | 0.1005 |
| Z_12 | 0.9600 | 0.9627 | 0.9632 | 0.9620 | 0.0014 | 0.1421 | 0.0781 |

Raw all-seed values: [h1a_layer_predictability_all_seeds.csv](h1a_layer_predictability_all_seeds.csv).

## 6. H1-B — direct alignment

Direct alignment uses per-sample cosine, absolute cosine, sample-wise projection energy `cosine²`, and linear CKA, separately on dev and val. `Z_0` is image-independent before the visual transformer, so its centered CKA is undefined and is recorded as NA; this is an architectural baseline, not a missing computation. Values are in [h1b_direct_alignment_all_seeds.csv](h1b_direct_alignment_all_seeds.csv).

## 7. H1-C — layer progression

| seed | early Z1-4 | middle Z5-8 | late Z9-12 | late-early | final-early | alignment late-early |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9236 | 0.9087 | 0.9394 | 0.0158 | 0.0364 | -0.1892 |
| 1 | 0.9304 | 0.9175 | 0.9452 | 0.0148 | 0.0322 | -0.1804 |
| 2 | 0.9345 | 0.9210 | 0.9465 | 0.0120 | 0.0287 | -0.1801 |

The core visualizations are [Figure 1](../outputs/pdf_representation_diagnosis/figures/figure1_depth_vs_cloth_predictability.png), [Figure 2](../outputs/pdf_representation_diagnosis/figures/figure2_depth_vs_identity_auc.png), and [Figure 3](../outputs/pdf_representation_diagnosis/figures/figure3_dual_axis_information_flow.png).

## 8. Layer-wise identity discrimination

| layer | diff-positive cosine | negative cosine | D_gap | ROC-AUC |
| --- | --- | --- | --- | --- |
| Z_0 | 1.0000 | 1.0000 | -0.0000 | 0.5000 |
| Z_1 | 0.9986 | 0.9986 | 0.0000 | 0.5120 |
| Z_2 | 0.9987 | 0.9987 | 0.0000 | 0.5149 |
| Z_3 | 0.9978 | 0.9977 | 0.0001 | 0.5153 |
| Z_4 | 0.9952 | 0.9951 | 0.0001 | 0.5100 |
| Z_5 | 0.9920 | 0.9917 | 0.0002 | 0.5181 |
| Z_6 | 0.9883 | 0.9879 | 0.0004 | 0.5210 |
| Z_7 | 0.9883 | 0.9877 | 0.0006 | 0.5381 |
| Z_8 | 0.9903 | 0.9897 | 0.0006 | 0.5474 |
| Z_9 | 0.9933 | 0.9926 | 0.0007 | 0.5826 |
| Z_10 | 0.9955 | 0.9937 | 0.0018 | 0.7821 |
| Z_11 | 0.9855 | 0.9624 | 0.0231 | 0.9793 |
| Z_12 | 0.9176 | 0.6534 | 0.2641 | 0.9991 |

Identity positives are fixed same-ID different-clothes A-C and B-C pairs; negatives are the shared different-ID manifest. Full values: [layer_identity_discrimination_all_seeds.csv](layer_identity_discrimination_all_seeds.csv).

## 9. H1 verdict

**MODERATE SUPPORT**. Q1: final CLS is easier to predict `F_cloth` than the early Z1-4 average, with final-minus-early val cosine `0.0364, 0.0322, 0.0287` across seeds. Q2: the direction is `3/3` seeds positive. This supports the bounded statement that later PDF identity-stream representations contain more information predictive of the PDF-defined cloth-related component; it does not prove ground-truth clothing leakage.

## 10. H2-A — `F_cloth` identity verification

| seed | positive mean | negative mean | D_gap | ROC-AUC | PR-AUC | EER |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9483 | 0.9022 | 0.0461 | 0.9734 | 0.9863 | 0.0825 |
| 1 | 0.9532 | 0.9112 | 0.0419 | 0.9690 | 0.9846 | 0.0875 |
| 2 | 0.9553 | 0.9156 | 0.0397 | 0.9726 | 0.9860 | 0.0828 |

Same-clothes A-B and separate A-C/B-C results for `F_visual`, `F_cloth`, and residuals are in [representation_verification_all_seeds.csv](representation_verification_all_seeds.csv). Q3: `F_cloth` cross-clothes verification AUC is `0.9734, 0.9690, 0.9726` across the three seeds, so the evidence is `STRONG SUPPORT` under the stated 0.55/0.60/0.70 diagnostic bands.


**Table 3 — representation diagnostic (mean ± seed std).**


| representation | same A-B AUC | diff A-C/B-C AUC | A/C R1 | A/C mAP | A/B R1 | A/B mAP |
| --- | --- | --- | --- | --- | --- | --- |
| F_visual | 0.9996 ± 0.0000 | 0.9992 ± 0.0000 | 98.5212 ± 0.0808% | 0.9669 ± 0.0007 | 99.5800 ± 0.0235% | 0.9885 ± 0.0003 |
| F_cloth | 0.9936 ± 0.0009 | 0.9717 ± 0.0019 | 82.8681 ± 0.0679% | 0.7057 ± 0.0061 | 97.8497 ± 0.1241% | 0.9384 ± 0.0006 |
| F_residual(alpha=0.00) | NA | 0.9992 ± 0.0000 | 98.5212 ± 0.0808% | 0.9669 ± 0.0007 | NA | NA |
| F_residual(alpha=0.25) | NA | 0.9992 ± 0.0000 | 98.5514 ± 0.0749% | 0.9678 ± 0.0007 | NA | NA |
| F_residual(alpha=0.50) | NA | 0.9991 ± 0.0000 | 98.5916 ± 0.0679% | 0.9687 ± 0.0007 | NA | NA |
| F_residual(alpha=0.75) | NA | 0.9991 ± 0.0000 | 98.6620 ± 0.0498% | 0.9695 ± 0.0007 | NA | NA |
| F_residual(alpha=1.00) | NA | 0.9991 ± 0.0000 | 98.7023 ± 0.0493% | 0.9702 ± 0.0007 | NA | NA |
| F_residual(alpha=1.25) | NA | 0.9991 ± 0.0001 | 98.7123 ± 0.0310% | 0.9707 ± 0.0007 | NA | NA |
| F_residual(alpha=1.50) | NA | 0.9991 ± 0.0001 | 98.6922 ± 0.0582% | 0.9708 ± 0.0008 | NA | NA |

## 11. H2-B — `F_cloth` cross-clothes retrieval

| seed | F_visual R1 | F_cloth R1 | F_visual mAP | F_cloth mAP |
| --- | --- | --- | --- | --- |
| 0 | 98.51% | 82.96% | 0.9677 | 0.7099 |
| 1 | 98.43% | 82.83% | 0.9660 | 0.7101 |
| 2 | 98.63% | 82.81% | 0.9669 | 0.6971 |

Q4: `F_cloth` A-gallery/C-query retrieval is nontrivial relative to 1/150 identity chance (`0.67%`), with R1 `82.96%, 82.83%, 82.81%`. A-gallery/B-query controls and all residual retrieval rows are in [retrieval_all_seeds.csv](retrieval_all_seeds.csv).

## 12. H2-C — offline alpha sweep

| alpha | R1 mean ± std | mAP mean ± std | ROC-AUC mean ± std | D_gap mean ± std |
| --- | --- | --- | --- | --- |
| 0.00 | 98.52 ± 0.08% | 0.9669 ± 0.0007 | 0.9992 ± 0.0000 | 0.4432 ± 0.0101 |
| 0.25 | 98.55 ± 0.07% | 0.9678 ± 0.0007 | 0.9992 ± 0.0000 | 0.3692 ± 0.0105 |
| 0.50 | 98.59 ± 0.07% | 0.9687 ± 0.0007 | 0.9991 ± 0.0000 | 0.3084 ± 0.0102 |
| 0.75 | 98.66 ± 0.05% | 0.9695 ± 0.0007 | 0.9991 ± 0.0000 | 0.2597 ± 0.0096 |
| 1.00 | 98.70 ± 0.05% | 0.9702 ± 0.0007 | 0.9991 ± 0.0000 | 0.2212 ± 0.0089 |
| 1.25 | 98.71 ± 0.03% | 0.9707 ± 0.0007 | 0.9991 ± 0.0001 | 0.1908 ± 0.0081 |
| 1.50 | 98.69 ± 0.06% | 0.9708 ± 0.0008 | 0.9991 ± 0.0001 | 0.1667 ± 0.0074 |

The trade-off plot is [Figure 4](../outputs/pdf_representation_diagnosis/figures/figure4_alpha_cross_clothes_retrieval.png). Q5: the mean curve shows a shallow R1 peak at alpha=1.25 followed by a 0.020 pp drop at alpha=1.50, while mAP keeps rising to alpha=1.50; it is not a joint inverted-U. Q6: alpha=1 is the best grid point for seed2 only; it does not exceed the best point for the other seeds. Per-seed best-minus-1 deltas are reported above and in [diagnosis_decision.json](diagnosis_decision.json).

## 13. Over-suppression analysis

The preregistered signal is: best alpha `< 1.0`, alpha=1 lower than best by at least 0.3 percentage points R1 or 0.2 percentage points mAP, in at least 2/3 seeds. Result: **NOT SUPPORTED**. This is evidence about the frozen subtraction mechanism, not a training change.

## 14. Optional region analysis

Skipped. The environment exposes a parsing directory for PRCC TEST only and no reliable PRCC TRAIN parsing cache. No new parsing model was deployed, and no Figure 6 is generated. Consequently there is no basis here to claim head/limb/body-structure evidence in `F_cloth`.

## 15. Three-seed consistency

All core extraction, H1, H2, and alpha-sweep stages were run for seed0, seed1, and seed2. The report retains individual seed values rather than selecting a favorable seed.

| seed | F_cloth AUC | F_cloth A/C R1 | best alpha | best-1 R1 (pp) | best-1 mAP (pp) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.9734 | 82.96% | 1.50 | 0.015 | 0.062 |
| 1 | 0.9690 | 82.83% | 1.25 | 0.030 | 0.044 |
| 2 | 0.9726 | 82.81% | 1.00 | 0.000 | 0.000 |

## 16. Bootstrap

Core train-only bootstrap summaries use 10,000 resamples at the identity unit where defined: `F_cloth` cross-clothes D-gap/AUC, best-minus-alpha=1 retrieval deltas, and prototype identity retention.

| seed | metric | estimate | 95% CI | n ID |
| --- | --- | --- | --- | --- |
| 0 | F_cloth_diff_clothes_D_gap | 0.0460 | [0.0447, 0.0473] | 150 |
| 0 | F_cloth_diff_clothes_ROC_AUC | 0.9805 | [0.9768, 0.9841] | 150 |
| 0 | alpha_best_minus_1_mAP | 0.0006 | [0.0001, 0.0011] | 150 |
| 0 | alpha_best_minus_1_R1 | 0.0002 | [-0.0008, 0.0013] | 150 |
| 0 | F_cloth_identity_retention_prototype | 0.0431 | [0.0402, 0.0460] | 150 |
| 1 | F_cloth_diff_clothes_D_gap | 0.0419 | [0.0408, 0.0430] | 150 |
| 1 | F_cloth_diff_clothes_ROC_AUC | 0.9777 | [0.9726, 0.9820] | 150 |
| 1 | alpha_best_minus_1_mAP | 0.0005 | [0.0001, 0.0008] | 150 |
| 1 | alpha_best_minus_1_R1 | 0.0004 | [-0.0003, 0.0011] | 150 |
| 1 | F_cloth_identity_retention_prototype | 0.0390 | [0.0364, 0.0417] | 150 |
| 2 | F_cloth_diff_clothes_D_gap | 0.0395 | [0.0383, 0.0406] | 150 |
| 2 | F_cloth_diff_clothes_ROC_AUC | 0.9780 | [0.9731, 0.9823] | 150 |
| 2 | alpha_best_minus_1_mAP | 0.0000 | [0.0000, 0.0000] | 150 |
| 2 | alpha_best_minus_1_R1 | 0.0000 | [0.0000, 0.0000] | 150 |
| 2 | F_cloth_identity_retention_prototype | 0.0368 | [0.0343, 0.0394] | 150 |

Full machine-readable output: [bootstrap_summary.csv](bootstrap_summary.csv).

## 17. H2 verdict

**STRONG SUPPORT**. Q7: evidence for PDF-defined representation leakage is **MODERATE SUPPORT**, based on the identity-disjoint layer probe and progression. Q8: evidence for over-suppression is **NOT SUPPORTED** under the fixed alpha rule. Q3/Q4 are the H2 identity-retention checks; their conclusions are limited to identity-correlated information in `F_cloth`.

## 18. Final decision matrix

| case | condition | interpretation | route |
| --- | --- | --- | --- |
| A | H1 supported + H2 supported | leakage and useful identity information both present | Separated Attention + Structure Preservation |
| B | H1 supported + H2 not supported | cloth component enters identity representation progressively | Separated Attention only |
| C | H1 not supported + H2 supported | suppression risks removing identity-correlated information | Structure Preservation only |
| D | H1 not supported + H2 not supported | hypothesized bottlenecks lack evidence | Neither — rethink bottleneck |

Observed decision: **MODERATE SUPPORT / STRONG SUPPORT -> A. Separated Attention + Structure Preservation**.

## 19. Recommended next method

Q9: **A. Separated Attention + Structure Preservation**. This recommendation follows only from the TRAIN dev/val and TRAIN diagnostic evidence above; PRCC TEST was not used to select it.

## 20. Limitations

- `F_cloth` is the PDF-defined `com_proj` path, not a ground-truth clothing annotation.
- The caption-conditioned cloth component depends on the frozen caption/cache path and the eval-mode BN running statistics used to make extraction deterministic.
- H1 probes use only the frozen 888-image selection; H2 uses full TRAIN pairs/retrieval, so the populations are intentionally different and explicitly reported.
- AUC/retrieval evidence is correlational and cannot by itself establish causal information flow.
- Region analysis is unavailable without a TRAIN parsing cache.
- Bootstrap CIs reflect identity resampling; they do not remove all image-level dependence within identity.
- No formal PRCC TEST benchmark was run in this phase.


## Artifact index

- Audit: [pdf_representation_audit.md](pdf_representation_audit.md)
- Checkpoint manifest: [baseline_checkpoint_manifest.json](baseline_checkpoint_manifest.json)
- Figures: `../outputs/pdf_representation_diagnosis/figures/`
- Per-seed features: `../outputs/pdf_representation_diagnosis/seed{0,1,2}/`

