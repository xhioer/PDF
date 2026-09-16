# PDF_visual_relation_confirmatory_report

## Executive Summary

- Confirmatory scope: seed0 R00/R02/R08 from the prior round plus new seed1/seed2 R00/R02/R08; all six new runs completed before this report was generated.
- Verdict 1 — visual hard-negative relation: **NO-GO**.
- Verdict 2 — joint semantic reliability: **NOT SUPPORTED**.
- Verdict 3 — next direction: **C — Stop the current relation-triplet formulation and move to visual identity structure preservation or stronger visual relation modeling.**
- The primary decision uses same-seed matched deltas across three seeds; seed0 rank alone is not used for method selection.

Operational note: the R02 primary criteria were operationalized exactly as specified (Diff R1 mean ≥ +0.30 pp or Diff mAP mean ≥ +0.20 pp, at least 2/3 positive). ‘Clearly negative’ was conservatively treated as a mean below −0.30 pp on either primary metric; raw values are reported below.

## 1. Diagnostic bug audit and repair

The post-analysis now reads independent copies of `semantic_neg[:, 0]` and `hybrid_neg[:, 0]` and records the graph-key mapping. The first 100 fixed anchors have 100% top-1 agreement, but the full train graph has only 97.7816% top-1 agreement and mean top-20 Jaccard 0.334874; this explains why the previous first-100 top-1 aggregates were identical without implying graph aliasing.

Diagnostic audit: `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/reports/semantic_hybrid_diagnostic_fix.md`

## 2. Frozen inputs and implementation audit

- branch/worktree: `pdf-crossclothes-relation-confirm` / `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm`
- source commit used by formal runs: recorded in each `run_provenance.json`; core PDF hashes remain unchanged from base.
- implementation audit: `pass`
- V0 checkpoint SHA256: `25c4a763fa8fab0b70e69207969312fddbb2e547f27281b83e20741c8f3272a3` (expected `25c4a763fa8fab0b70e69207969312fddbb2e547f27281b83e20741c8f3272a3`)
- P2 cache SHA256: `cd0aded0585c6af9f65451a7a05cb7a088601057600533fb2efd151d1275f20d`
- config consistency gate: `pass` with `0` non-allowed differences.
- relation graph manifest frozen/train-only/test_data_used: `True` / `True` / `False`
- relation graph ledger SHA256 sidecar: `70eaabec2641167ce12c1b06ecd134338becf7ccf25b018a73a4cb3f24310502  relation_graph_hashes.json`
- no test-adaptive checkpoint selection was used for relation mining.

## 3. Confirmatory protocol and six-run completion

The confirmatory runner reused the frozen visual-hard top-20 negatives, A-C/B-C same-ID cross-clothes positives, R_joint formula and active-edge mean normalization. No caption generation, new semantic encoding, graph re-mining, backbone change, margin change, lambda search, or inference change was introduced.

| Seed | Run | Status | Diff R1 | Diff mAP | Same R1 | Same mAP |
|---|---|---|---|---|---|---|
| 0 | R00 | complete | 64.352 | 62.146 | 99.871 | 98.139 |
| 0 | R02 | complete | 64.719 | 62.305 | 99.897 | 98.094 |
| 0 | R08 | complete | 64.663 | 62.350 | 99.897 | 98.099 |
| 1 | R00 | complete | 65.058 | 62.508 | 99.871 | 98.230 |
| 1 | R02 | complete | 65.086 | 62.640 | 99.897 | 98.245 |
| 1 | R08 | complete | 64.776 | 62.631 | 99.897 | 98.244 |
| 2 | R00 | complete | 66.526 | 62.215 | 99.897 | 98.016 |
| 2 | R02 | complete | 65.679 | 62.151 | 99.871 | 98.002 |
| 2 | R08 | complete | 65.651 | 62.152 | 99.871 | 97.999 |

Queue status: `complete`; completed tasks: `6/6`; elapsed hours: `7.366`.

## 4. Full Same/Different Clothes image-only metrics

| Seed | Run | Status | Diff R1 | Diff R5 | Diff R10 | Diff R20 | Diff mAP | Same R1 | Same R5 | Same R10 | Same R20 | Same mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | R00 | complete | 64.352 | 74.316 | 78.182 | 81.061 | 62.146 | 99.871 | 99.923 | 99.923 | 99.948 | 98.139 |
| 0 | R02 | complete | 64.719 | 74.485 | 78.182 | 80.864 | 62.305 | 99.897 | 99.923 | 99.923 | 99.923 | 98.094 |
| 0 | R08 | complete | 64.663 | 74.626 | 77.985 | 80.892 | 62.350 | 99.897 | 99.923 | 99.923 | 99.923 | 98.099 |
| 1 | R00 | complete | 65.058 | 74.541 | 77.505 | 79.876 | 62.508 | 99.871 | 99.923 | 99.923 | 99.948 | 98.230 |
| 1 | R02 | complete | 65.086 | 74.344 | 77.392 | 80.102 | 62.640 | 99.897 | 99.923 | 99.923 | 99.923 | 98.245 |
| 1 | R08 | complete | 64.776 | 74.316 | 77.364 | 79.932 | 62.631 | 99.897 | 99.923 | 99.923 | 99.923 | 98.244 |
| 2 | R00 | complete | 66.526 | 74.033 | 77.194 | 79.989 | 62.215 | 99.897 | 99.923 | 99.923 | 99.923 | 98.016 |
| 2 | R02 | complete | 65.679 | 73.666 | 77.251 | 80.412 | 62.151 | 99.871 | 99.923 | 99.923 | 99.923 | 98.002 |
| 2 | R08 | complete | 65.651 | 73.779 | 77.448 | 80.271 | 62.152 | 99.871 | 99.923 | 99.923 | 99.923 | 97.999 |

All test metrics above come from epoch50 final image-only evaluation. TEST labels are used only by the evaluator at evaluation time; no TEST identity labels/features/captions/pair metadata entered graph construction, mining, training, or checkpoint selection.

## 5. Three-seed matched deltas

| Transition | Metric | Seed deltas (pp) | Mean (pp) | Std (pp) | Median (pp) | >0 seeds |
|---|---|---|---|---|---|---|
| R02 − R00 | diff_R1 | 0.367, 0.028, -0.847 | -0.151 | 0.626 | 0.028 | 2 |
| R02 − R00 | diff_mAP | 0.159, 0.132, -0.064 | 0.076 | 0.122 | 0.132 | 2 |
| R02 − R00 | same_R1 | 0.026, 0.026, -0.026 | 0.009 | 0.030 | 0.026 | 2 |
| R02 − R00 | same_mAP | -0.044, 0.015, -0.014 | -0.014 | 0.029 | -0.014 | 1 |
| R08 − R00 | diff_R1 | 0.310, -0.282, -0.875 | -0.282 | 0.593 | -0.282 | 1 |
| R08 − R00 | diff_mAP | 0.205, 0.124, -0.063 | 0.088 | 0.137 | 0.124 | 2 |
| R08 − R00 | same_R1 | 0.026, 0.026, -0.026 | 0.009 | 0.030 | 0.026 | 2 |
| R08 − R00 | same_mAP | -0.040, 0.014, -0.017 | -0.014 | 0.027 | -0.017 | 1 |
| R08 − R02 | diff_R1 | -0.056, -0.310, -0.028 | -0.132 | 0.155 | -0.056 | 0 |
| R08 − R02 | diff_mAP | 0.045, -0.009, 0.001 | 0.013 | 0.029 | 0.001 | 2 |
| R08 − R02 | same_R1 | 0.000, 0.000, 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| R08 − R02 | same_mAP | 0.004, -0.001, -0.003 | 0.000 | 0.004 | -0.001 | 1 |

### R02 versus R00

R02 is judged against the registered thresholds above, with both primary metrics shown rather than selecting the better seed0 rank.

### R08 versus R02

R08 receives reliability support only if its matched Diff R1 mean is at least +0.15 pp or its Diff mAP mean is at least +0.10 pp, with at least 2/3 positive seeds.

## 6. Relation diagnostics and effective loss scale

Final-epoch relation diagnostics for the confirmatory R02/R08 runs are retained in each run’s `metrics.csv`; the compact table below shows the final row.

| Seed | Run | Pos cos | Visual-hard cos | Margin | Active hinge | Relation grad/total |
|---|---|---|---|---|---|---|
| 0 | R02 | 0.8852 | 0.8513 | 0.0605 | 0.9999 | 0.0151 |
| 0 | R08 | 0.8841 | 0.8498 | 0.0610 | 0.9999 | 0.0154 |
| 1 | R02 | 0.8805 | 0.8422 | 0.0653 | 0.9998 | 0.0156 |
| 1 | R08 | 0.8793 | 0.8395 | 0.0661 | 0.9998 | 0.0159 |
| 2 | R02 | 0.8857 | 0.8554 | 0.0627 | 0.9999 | 0.0149 |
| 2 | R08 | 0.8848 | 0.8523 | 0.0632 | 0.9999 | 0.0153 |

R08 raw/normalized R_joint weight distributions and all per-epoch loss/gradient fields are stored in its `metrics.csv`; active-edge mean normalization is verified by the preflight and run provenance artifacts.

## 7. Pure visual mechanism evidence

The ‘before’ values use the frozen V0 train visual features on the first 100 fixed train anchors; the ‘after’ values use seed1 epoch50 final diagnostics. This is train-only mechanism evidence and is not a test metric.

{
  "before_positive_cosine": 0.9048967957496643,
  "before_positive_negative_margin": 0.01640123873949051,
  "before_visual_hard_negative_cosine": 0.8884955644607544,
  "fixed_anchor_count": 100,
  "runs": {
    "R02": {
      "after_positive_cosine": 0.8805355293284483,
      "after_positive_negative_margin": 0.0653220538008697,
      "after_visual_hard_negative_cosine": 0.8422150611877441,
      "seed": 1
    },
    "R08": {
      "after_positive_cosine": 0.8792651896107241,
      "after_positive_negative_margin": 0.06610111795998147,
      "after_visual_hard_negative_cosine": 0.8394923806190491,
      "seed": 1
    }
  }
}

## 8. Semantic-confuser before/after analysis

{
  "feature_definition": "raw projected CLS encode_image, L2 normalized; no flip",
  "fixed_anchor_count": 100,
  "graph_relation_index": "outputs/crossclothes_relation_confirm/relation_graph/relation_index.npz",
  "hybrid_top1": {
    "count": 300,
    "mean_delta": -0.0284915562470754,
    "mean_rank_after": 221.66333333333333,
    "mean_rank_before": 233.68,
    "mean_similarity_after": 0.8321087954441706,
    "mean_similarity_before": 0.860600351691246,
    "rank_improved_fraction": 0.32
  },
  "hybrid_top1_graph_key": "hybrid_neg[:, 0]",
  "missing": [],
  "rows": 600,
  "semantic_confusers": "frozen graph semantic_top1 and hybrid_top1 different-ID train negatives",
  "semantic_top1": {
    "count": 300,
    "mean_delta": -0.0284915562470754,
    "mean_rank_after": 221.66333333333333,
    "mean_rank_before": 233.68,
    "mean_similarity_after": 0.8321087954441706,
    "mean_similarity_before": 0.860600351691246,
    "rank_improved_fraction": 0.32
  },
  "semantic_top1_graph_key": "semantic_neg[:, 0]",
  "test_data_used": false,
  "top1_arrays_independent": true,
  "top1_exact_match_all": 17499,
  "top1_exact_match_first100": 100,
  "top1_exact_match_rate_all": 0.9778162717925794,
  "top1_exact_match_rate_first100": 1.0
}

The diagnostic is limited to fixed train anchors and frozen graph negatives. No claim is made that semantic hard-negative mining is generally ineffective; this round only uses it to audit the prior readout and does not train a semantic or hybrid method.

## 9. Bootstrap and uncertainty

Per-query evaluation outputs were not present in the runner artifacts, so the requested 10,000-query bootstrap CI was not computed. The primary uncertainty assessment is the pre-registered three-seed matched effect (mean, sample standard deviation, median, and positive-seed count). No pooled query-level independence assumption was substituted.

## 10. Runtime and failure cases

- two-device queue: max concurrent jobs 2; max one job per device; dynamic queue; 12-hour gate.
- queue elapsed: `7.366` hours; individual elapsed times are recorded in `run_summary.json` for each run.
- failure cases: none among the six formal runs; any AMP overflow counts remain diagnostics from the unchanged scaler and did not prevent successful final checkpoints.
- report readiness: **Ready within reviewed scope** for the completed six-run confirmatory study; the prior 24-run RCHRL-V1 experiment remains a separate 16/24 study and is not silently relabeled complete.

## 11. Final GO / NO-GO

### Verdict 1 — VISUAL HARD-NEGATIVE RELATION: **NO-GO**

R02 matched effect: Diff R1 mean `-0.151` pp, Diff mAP mean `0.076` pp; positive-seed counts are `2` and `2` respectively.

### Verdict 2 — SEMANTIC RELIABILITY: **NOT SUPPORTED**

R08 − R02 matched effect: Diff R1 mean `-0.132` pp, Diff mAP mean `0.013` pp; positive-seed counts are `0` and `2` respectively.

### Verdict 3 — NEXT RESEARCH DIRECTION

C — Stop the current relation-triplet formulation and move to visual identity structure preservation or stronger visual relation modeling.

## 12. Reproducibility artifacts

- config: `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/reports/config_diff_seed0_vs_confirm.json`
- preflight: `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/reports/confirm_preflight.json`
- diagnostic fix: `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/reports/semantic_hybrid_diagnostic_fix.md`
- fixed graph: `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs/crossclothes_relation_confirm/relation_graph`
- confirm queue status: `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs/crossclothes_relation_confirm/confirmatory_queue_status.json`
- finalizer status: `/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/reports/confirmatory_finalize_status.json`
