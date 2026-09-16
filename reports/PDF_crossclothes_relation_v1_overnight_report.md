# PDF_crossclothes_relation_v1 overnight report

结论先行：本报告状态为 **Needs revision**；24-run completion 为 **16/24**，科学结论分类为 **E**。

## 1. Implementation audit

```json
{
  "audit_status": "pass",
  "base_commit": "4eb619dca59a922fd28bc0891e8c3066e7186841",
  "branch": "pdf-crossclothes-relation-v1",
  "core_files_changed_since_base": [],
  "dynamic_remining": false,
  "forbidden_feature_code_hits": {},
  "image_only_inference_unchanged": true,
  "p2_cache_lines": 17896,
  "p2_cache_sha256": "cd0aded0585c6af9f65451a7a05cb7a088601057600533fb2efd151d1275f20d",
  "p2_validation_passed": true,
  "source_commit": "dce95bec4a93ff27ea06e01f3c9e50de91234fbf",
  "test_data_in_relation_graph_or_mining": false,
  "worktree": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1"
}
```

冻结核心文件相对 base commit 无改动；新增 runner/graph/sampler 位于独立 branch/worktree。原 PDF `train.py`、`models/clip_model.py`、`test.py` 与 inference 输出 shape 保持不变。

## 2. V0 mining checkpoint provenance

```json
{
  "checkpoint_path": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/v0_mining/epoch50_final.verified.pth",
  "checkpoint_sha256": "25c4a763fa8fab0b70e69207969312fddbb2e547f27281b83e20741c8f3272a3",
  "epoch": 50,
  "no_test_adaptive_checkpoint_selection_was_used_for_relation_mining": true,
  "provenance_correction": "verified copy created with the exact launch commit; model state unchanged",
  "raw_checkpoint_path": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/v0_mining/epoch50_final.pth",
  "raw_checkpoint_sha256": "16483fdbf93e3d1c9539f17020c7d8401241070ac4499bdce8c75510dc7738c7",
  "source_commit": "8745c8adfa035bcfebe88b6653faedd88e529080",
  "source_commit_at_launch": "8745c8adfa035bcfebe88b6653faedd88e529080",
  "test_adaptive_checkpoint_selection": false,
  "train_only": true,
  "training_seed": 0
}
```

明确记录：no test-adaptive checkpoint selection was used for relation mining。epoch50 final 是主 mining checkpoint；任何 best-test checkpoint 仅可作为辅助，不进入 graph。

## 3. Relation graph construction and SHA256

```json
{
  "frozen": true,
  "graph_version": "RCHRL-V1-frozen-train-graph-1",
  "hardness_definition": "per-anchor rank normalization K=20; stable path hash tie-break",
  "hybrid_definition": "0.5 H_visual_rank + 0.5 H_sem_rank over visual/semantic union, final rank top20",
  "matched_random_definition": "different-ID controls matched hierarchically to hybrid negative camera, clothes state, and P2 overlap indicators",
  "p2_cache_sha256": "cd0aded0585c6af9f65451a7a05cb7a088601057600533fb2efd151d1275f20d",
  "positive_definition": "same-ID A-C and B-C directed relations; A-B excluded",
  "semantic_allowed_attributes": [
    "gender",
    "hair_color",
    "hair_length",
    "body_build"
  ],
  "semantic_disallowed_fields": [
    "S2",
    "S4",
    "raw_model_output",
    "viewpoint",
    "face_visibility",
    "occlusion",
    "image_quality",
    "upper_clothing",
    "lower_clothing",
    "shoes",
    "accessories"
  ],
  "semantic_serialization": {
    "phrase_count": 13,
    "phrase_template": "A person with {attribute}: {value}.",
    "serialization": "mean of valid per-attribute phrases",
    "unknown_values_encoded": false
  },
  "test_data_used": false,
  "train_id_count": 150,
  "train_image_count": 17896,
  "train_only": true,
  "v0_checkpoint_sha256": "25c4a763fa8fab0b70e69207969312fddbb2e547f27281b83e20741c8f3272a3"
}
```

Hash ledger（graph freeze 后未重新 mining）：
- `hybrid_hard_negative_edges.jsonl`: `60d2077ad7274e29ad315afa556706496feb9025e453797890c794fd298c7a1a`
- `matched_random_negative_edges.jsonl`: `a613911687b32cd888e44d928c98fe0314c55ad4eb7d168d284baf42d5a693d4`
- `positive_edges.jsonl`: `cfeb9713a12fedc6504c035a768442c6ecb4364920f095e92e0982b503113761`
- `relation_graph_manifest.json`: `328c3ec1302a1019b064afec2741989ceea7f5e82c9a8f394566cfd1e6e7c9be`
- `relation_graph_stats.json`: `6953d890031f22a33fce2440c2f8850fd1dec43686a6158237ef0e225312ec31`
- `relation_index.npz`: `33d0ebaf424e88992e35d6f3c4dbc0640d2320f8215609230bcf77620f0a7221`
- `semantic_features.npy`: `74f1df745fc6d8b53d5c55cf7404dad952825559a842eb73ad28deaff9adf320`
- `semantic_hard_negative_edges.jsonl`: `e02e7c3afd8795469cb976879aebbddabe25fbb4fcea4ff3ca33ce7b394fbf23`
- `train_image_manifest.jsonl`: `8f3eba7902b55942a981a4a183f18bc2bf329a4a1a9169f2de33cd9e3e5d4a8c`
- `visual_features.npy`: `ce006d511ea494eea0ef64f43b28f47e2bb3724d40728584f3d471393268ea43`
- `visual_hard_negative_edges.jsonl`: `705e0a8d65adb0babcafb25af11bc8a89f0c2f179a88f8caad3d3d46f727c961`

独立 train-index validation：
```json
{
  "checks": {
    "feature_norms": {
      "semantic_nonzero_l2": true,
      "semantic_zero_vectors": 4,
      "visual_l2": true
    },
    "hybrid_neg": {
      "different_id": true,
      "range_valid": true,
      "shape": [
        17896,
        20
      ],
      "unique_per_anchor": true
    },
    "matched_neg": {
      "different_id": true,
      "range_valid": true,
      "shape": [
        17896,
        20
      ],
      "unique_per_anchor": true
    },
    "positive": {
      "directed_count": 1002080,
      "same_id_different_clothes_ac_bc_only": true
    },
    "semantic_neg": {
      "different_id": true,
      "range_valid": true,
      "shape": [
        17896,
        20
      ],
      "unique_per_anchor": true
    },
    "visual_neg": {
      "different_id": true,
      "range_valid": true,
      "shape": [
        17896,
        20
      ],
      "unique_per_anchor": true
    }
  },
  "experiment": "RCHRL-V1",
  "graph_dir": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
  "manifest_frozen": true,
  "manifest_test_data_used": false,
  "manifest_train_only": true,
  "payload_sha256": {
    "hybrid_hard_negative_edges.jsonl": "60d2077ad7274e29ad315afa556706496feb9025e453797890c794fd298c7a1a",
    "matched_random_negative_edges.jsonl": "a613911687b32cd888e44d928c98fe0314c55ad4eb7d168d284baf42d5a693d4",
    "positive_edges.jsonl": "cfeb9713a12fedc6504c035a768442c6ecb4364920f095e92e0982b503113761",
    "relation_graph_manifest.json": "328c3ec1302a1019b064afec2741989ceea7f5e82c9a8f394566cfd1e6e7c9be",
    "relation_graph_stats.json": "6953d890031f22a33fce2440c2f8850fd1dec43686a6158237ef0e225312ec31",
    "relation_index.npz": "33d0ebaf424e88992e35d6f3c4dbc0640d2320f8215609230bcf77620f0a7221",
    "semantic_features.npy": "74f1df745fc6d8b53d5c55cf7404dad952825559a842eb73ad28deaff9adf320",
    "semantic_hard_negative_edges.jsonl": "e02e7c3afd8795469cb976879aebbddabe25fbb4fcea4ff3ca33ce7b394fbf23",
    "train_image_manifest.jsonl": "8f3eba7902b55942a981a4a183f18bc2bf329a4a1a9169f2de33cd9e3e5d4a8c",
    "visual_features.npy": "ce006d511ea494eea0ef64f43b28f47e2bb3724d40728584f3d471393268ea43",
    "visual_hard_negative_edges.jsonl": "705e0a8d65adb0babcafb25af11bc8a89f0c2f179a88f8caad3d3d46f727c961"
  },
  "verification": "verify_frozen_graph plus train-index structural validation passed"
}
```

## 4. Raw reliability distributions

```json
{
  "R_agr": {
    "count": 1002080,
    "histogram": {
      "[0.000,0.100)": 658,
      "[0.100,0.200)": 0,
      "[0.200,0.300)": 19152,
      "[0.300,0.400)": 786,
      "[0.400,0.500)": 0,
      "[0.500,0.600)": 89830,
      "[0.600,0.700)": 3598,
      "[0.700,0.800)": 288180,
      "[0.800,0.900)": 0,
      "[0.900,1.000)": 599876
    },
    "max": 1.0,
    "mean": 0.866572196497951,
    "median": 1.0,
    "min": 0.0,
    "q1": 0.75,
    "q3": 1.0,
    "std": 0.18637757386513068,
    "unique_count": 7,
    "zero_rate": 0.0006566342008622067
  },
  "R_conf": {
    "count": 1002080,
    "histogram": {
      "[0.000,0.100)": 340,
      "[0.100,0.200)": 290,
      "[0.200,0.300)": 6324,
      "[0.300,0.400)": 24784,
      "[0.400,0.500)": 34556,
      "[0.500,0.600)": 97880,
      "[0.600,0.700)": 273936,
      "[0.700,0.800)": 297834,
      "[0.800,0.900)": 240846,
      "[0.900,1.000)": 25290
    },
    "max": 1.0,
    "mean": 0.6965983473257317,
    "median": 0.7285533905932737,
    "min": 0.0,
    "q1": 0.625,
    "q3": 0.8017766952966369,
    "std": 0.1323990526938416,
    "unique_count": 28,
    "zero_rate": 0.00033929426792272076
  },
  "R_joint": {
    "count": 1002080,
    "histogram": {
      "[0.000,0.100)": 2004,
      "[0.100,0.200)": 26670,
      "[0.200,0.300)": 27688,
      "[0.300,0.400)": 93780,
      "[0.400,0.500)": 130848,
      "[0.500,0.600)": 129306,
      "[0.600,0.700)": 201654,
      "[0.700,0.800)": 203160,
      "[0.800,0.900)": 172140,
      "[0.900,1.000)": 14830
    },
    "max": 1.0,
    "mean": 0.6089683353301811,
    "median": 0.625,
    "min": 0.0,
    "q1": 0.49149756441743303,
    "q3": 0.75,
    "std": 0.1850936391394372,
    "unique_count": 99,
    "zero_rate": 0.0006566342008622067
  },
  "collapse_warning": {
    "R_agr": false,
    "R_conf": false,
    "R_joint": false
  }
}
```

R_conf 使用四个允许属性 confidence 的 geometric mean pair confidence；R_agr 对两端有效且 exact-equal 的 categorical values 求平均，unknown 不进入 denominator；R_joint=R_conf×R_agr。Identity-level 分布完整保存在 `relation_graph_stats.json`。

## 5. Normalized training-weight distributions and effective relation scale

每个正式 run 的 `metrics.csv` 逐 epoch 记录 raw/normalized weight mean/std/zero-rate、active-edge rate、raw relation loss、weighted relation loss、lambda contribution、total loss、gradient norm 与 ratio。active weight normalization 按 active edge mean 执行；zero edge 保持 0。

| Seed0 run | raw mean | raw std | raw zero | norm mean | active norm mean | active rate | weighted relation | lambda/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| R01 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.028159458929074867 | 0.0005562176640215072 |
| R02 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.23951516424157962 | 0.004712899844419912 |
| R03 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.030651127067274273 | 0.000605474804172368 |
| R04 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.1353578377528824 | 0.002669004576514403 |
| R05 | 0.6960672568461111 | 0.1338846215135188 | 0.00034594095940959407 | 0.9996540583592816 | 0.9999999993184554 | 0.9996540590405905 | 0.1341422455887073 | 0.0026450300197429764 |
| R06 | 0.8681196187884028 | 0.18403393098026138 | 0.0006342250922509225 | 0.9993657798117982 | 1.0000000049071613 | 0.999365774907749 | 0.1343729166911977 | 0.002649647382610371 |
| R07 | 0.609706792700778 | 0.18458142921296156 | 0.0006342250922509225 | 0.9993657747539606 | 0.9999999998461139 | 0.999365774907749 | 0.133380281996683 | 0.0026301531214782507 |
| R08 | 0.609706792700778 | 0.18458142921296156 | 0.0006342250922509225 | 0.9993657747539606 | 0.9999999998461139 | 0.999365774907749 | 0.237373038521552 | 0.004670801297572486 |
| R09 | 0.609706792700778 | 0.18458142921296156 | 0.0006342250922509225 | 0.9993657747539606 | 0.9999999998461139 | 0.999365774907749 | 0.02953744857706047 | 0.0005835181483298685 |
| R10 | 0.500473394761862 | 0.30318484383673705 | 0.04946955719557196 | 0.9505304431197374 | 1.0000000003317193 | 0.950530442804428 | 0.12722028530721735 | 0.0025089301849368314 |
| R11 | 0.30511381036487156 | 0.21375882316052505 | 0.050046125461254615 | 0.9499538748850918 | 1.0000000003645928 | 0.9499538745387454 | 0.12516365667660737 | 0.002468292921011738 |
| R12 | 0.30511381036487156 | 0.21375882316052505 | 0.050046125461254615 | 0.9499538748850918 | 1.0000000003645928 | 0.9499538745387454 | 0.14072718729722103 | 0.0013889716203561624 |
| R13 | 0.30511381036487156 | 0.21375882316052505 | 0.050046125461254615 | 0.9499538748850918 | 1.0000000003645928 | 0.9499538745387454 | 0.10906618841789745 | 0.004295348023954518 |
| R14 | 0.609706792700778 | 0.18458142921296156 | 0.0006342250922509225 | 0.9993657747539606 | 0.9999999998461139 | 0.999365774907749 | 0.053093343986906245 | 0.001047982215044532 |
| R15 | 0.500473394761862 | 0.30318484383673705 | 0.04946955719557196 | 0.9505304431197374 | 1.0000000003317193 | 0.950530442804428 | 0.10555784277810382 | 0.0020822464356137047 |

## 6. Metadata-matched random negative validation

```json
{
  "hybrid": {
    "body_build_overlap": 0.8702279839070183,
    "camera_pair_distribution": {
      "A->A": 44227,
      "A->B": 25330,
      "A->C": 49483,
      "B->A": 37473,
      "B->B": 27033,
      "B->C": 41834,
      "C->A": 46497,
      "C->B": 28596,
      "C->C": 57447
    },
    "clothes_pair_distribution": {
      "different_clothes->different_clothes": 57447,
      "different_clothes->same_clothes": 75093,
      "same_clothes->different_clothes": 91317,
      "same_clothes->same_clothes": 134063
    },
    "gender_overlap": 0.9864047831917747,
    "hair_color_overlap": 0.9158722619579794,
    "hair_length_overlap": 0.910513522574877,
    "mean_semantic_cosine": 0.9986133635024115,
    "mean_visual_cosine": 0.7912045663230164,
    "negative_count": 357920,
    "negative_count_per_anchor": {
      "max": 20,
      "mean": 20.0,
      "min": 20
    },
    "same_clothes_negative_ratio": 0.584365221278498
  },
  "matched_random": {
    "body_build_overlap": 0.8698312472060795,
    "camera_pair_distribution": {
      "A->A": 44227,
      "A->B": 25330,
      "A->C": 49483,
      "B->A": 37473,
      "B->B": 27033,
      "B->C": 41834,
      "C->A": 46497,
      "C->B": 28596,
      "C->C": 57447
    },
    "clothes_pair_distribution": {
      "different_clothes->different_clothes": 57447,
      "different_clothes->same_clothes": 75093,
      "same_clothes->different_clothes": 91317,
      "same_clothes->same_clothes": 134063
    },
    "gender_overlap": 0.9864047831917747,
    "hair_color_overlap": 0.9158666741171212,
    "hair_length_overlap": 0.9105107286544479,
    "match_level_distribution": {
      "anchor_camera+negative_camera+negative_clothes_state+gender": 2,
      "anchor_camera+negative_camera+negative_clothes_state+gender+hair_color+hair_length": 141,
      "anchor_camera+negative_camera+negative_clothes_state+gender+hair_color+hair_length+body_build": 357777
    },
    "mean_semantic_cosine": 0.9985634299615463,
    "mean_visual_cosine": 0.7003080833197026,
    "negative_count": 357920,
    "negative_count_per_anchor": {
      "max": 20,
      "mean": 20.0,
      "min": 20
    },
    "same_clothes_negative_ratio": 0.584365221278498
  },
  "semantic": {
    "body_build_overlap": 0.9993322530174341,
    "camera_pair_distribution": {
      "A->A": 43101,
      "A->B": 28110,
      "A->C": 47829,
      "B->A": 38892,
      "B->B": 27408,
      "B->C": 40040,
      "C->A": 47773,
      "C->B": 30814,
      "C->C": 53953
    },
    "clothes_pair_distribution": {
      "different_clothes->different_clothes": 53953,
      "different_clothes->same_clothes": 78587,
      "same_clothes->different_clothes": 87869,
      "same_clothes->same_clothes": 137511
    },
    "gender_overlap": 0.9967087617344658,
    "hair_color_overlap": 0.9866310907465355,
    "hair_length_overlap": 0.9957867679928476,
    "mean_semantic_cosine": 0.9997621065311244,
    "mean_visual_cosine": 0.7044937118954514,
    "negative_count": 357920,
    "negative_count_per_anchor": {
      "max": 20,
      "mean": 20.0,
      "min": 20
    },
    "same_clothes_negative_ratio": 0.6037606168976307
  },
  "visual": {
    "body_build_overlap": 0.7381845105051408,
    "camera_pair_distribution": {
      "A->A": 43899,
      "A->B": 29263,
      "A->C": 45878,
      "B->A": 35782,
      "B->B": 31996,
      "B->C": 38562,
      "C->A": 45089,
      "C->B": 32934,
      "C->C": 54517
    },
    "clothes_pair_distribution": {
      "different_clothes->different_clothes": 54517,
      "different_clothes->same_clothes": 78023,
      "same_clothes->different_clothes": 84440,
      "same_clothes->same_clothes": 140940
    },
    "gender_overlap": 0.9741729995529728,
    "hair_color_overlap": 0.8518635449262405,
    "hair_length_overlap": 0.8242596110862762,
    "mean_semantic_cosine": 0.9975119163947922,
    "mean_visual_cosine": 0.8714947789494459,
    "negative_count": 357920,
    "negative_count_per_anchor": {
      "max": 20,
      "mean": 20.0,
      "min": 20
    },
    "same_clothes_negative_ratio": 0.6117651989271345
  }
}
```

matched-random 每个 anchor 20 条 control，different-ID；match level、camera/clothes pair、same-clothes rate 与四个属性 overlap 已在上述 JSON 及 graph stats 中记录。

## 7. Visual/semantic/hybrid overlap and semantic-hard audit

```json
{
  "overlap": {
    "semantic_hybrid": 0.33487425614735894,
    "semantic_not_visual_top20_rate": 0.9970160929816718,
    "visual_hybrid": 0.3339785500314533,
    "visual_semantic": 0.001533828381345027
  },
  "semantic_confuser_analysis": {
    "interpretation": "semantic-hard is a semantic-nearest different-ID set; compare its visual cosine to visual-hard and its semantic cosine to visual-hard to test whether it adds visually non-nearest semantic confusers",
    "semantic_hard_is_visually_not_nearest_but_semantically_similar": true,
    "semantic_hard_mean_semantic_cosine": 0.9997621065311244,
    "semantic_hard_mean_visual_cosine": 0.7044937118954514,
    "visual_hard_mean_semantic_cosine": 0.9975119163947922,
    "visual_hard_mean_visual_cosine": 0.8714947789494459
  }
}
```

判定依据是 semantic-hard 的 semantic cosine 是否高于 visual-hard，同时 visual cosine 不必同样高；该判断不以 TEST 为依据。

## 8. Extreme hard-negative audit

`reports/extreme_hard_negative_audit.csv` 包含 1500 个 audit rows（前 100 个固定 train anchors 的 visual/semantic/hybrid top-5），含路径、person/camera/clothes metadata、两种 cosine 与 P2 属性。

## 9. Gradient sanity

```json
[
  {
    "amp_overflow_steps": 4,
    "finite": true,
    "gradient_finite_on_successful_updates": true,
    "gradient_finite_rate": 0.8620689655172413,
    "iterations": 29,
    "nonfinite_gradient_steps": 4,
    "optimizer_update_success": true,
    "path": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/sanity/R01",
    "phase": "sanity",
    "rows": [
      {
        "H_hybrid": 0.49838362377265405,
        "H_sem": 0.0,
        "H_visual": 0.0,
        "R_agr": 0.8671875082213303,
        "R_conf": 0.6919865341022097,
        "R_joint": 0.6050584275147011,
        "active_edge_rate": 1.0,
        "active_hinge_rate": 1.0,
        "amp_overflow_steps": 4,
        "batches": 29,
        "edge_repeat_rate": 0.0,
        "epoch": 1,
        "finite": true,
        "fixed_diagnostic_positive_cosine": 0.8490296006202698,
        "gradient_finite_rate": 0.8620689655172413,
        "hybrid_hard_negative_cosine": 0.7896530628204346,
        "images_per_sec": 45.604176457118776,
        "lambda_over_total": 0.0032330973330756715,
        "lambda_relation_loss": 0.028641007005654532,
        "matched_random_negative_cosine": 0.8114408254623413,
        "max_edge_repeats": 1,
        "negative_edge_coverage": 1.0,
        "nonfinite_gradient_steps": 4,
        "normalized_active_weight_mean": 1.0,
        "normalized_weight_mean": 1.0,
        "normalized_weight_std": 0.0,
        "positive_cosine": 0.829506438353966,
        "positive_edge_coverage": 1.0,
        "positive_negative_margin": 0.0135899409972902,
        "raw_relation_loss_before_weight": 0.28641007172650307,
        "raw_weight_mean": 1.0,
        "raw_weight_std": 0.0,
        "raw_weight_zero_rate": 0.0,
        "relation_diagnostic_anchor_count": 64,
        "relation_grad_over_total_grad": 0.0027245014632022936,
        "relation_gradient_norm": 0.5083624086122712,
        "sampled_negative_cosine": 0.8159164971318739,
        "sec_per_iter": 1.4033802377766575,
        "seconds": 40.69802689552307,
        "semantic_hard_negative_cosine": 0.7896530628204346,
        "successful_optimizer_steps": 25,
        "total_loss": 8.875638830250708,
        "unique_anchors": 1856,
        "unique_negative_images": 986,
        "unique_positive_images": 1798,
        "unique_relation_edges": 1856,
        "visual_backbone_gradient_norm": 186.5359265557651,
        "visual_hard_negative_cosine": 0.8256833553314209,
        "weighted_relation_loss_before_lambda": 0.2864100706988367
      }
    ],
    "run_id": "R01",
    "seed": 0,
    "unexpected_trainable_relation_parameters": 0
  },
  {
    "amp_overflow_steps": 4,
    "finite": true,
    "gradient_finite_on_successful_updates": true,
    "gradient_finite_rate": 0.8620689655172413,
    "iterations": 29,
    "nonfinite_gradient_steps": 4,
    "optimizer_update_success": true,
    "path": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/sanity/R04",
    "phase": "sanity",
    "rows": [
      {
        "H_hybrid": 0.49838362377265405,
        "H_sem": 0.38089836465901344,
        "H_visual": 0.3828833935589626,
        "R_agr": 0.8671875082213303,
        "R_conf": 0.6919865341022097,
        "R_joint": 0.6050584275147011,
        "active_edge_rate": 1.0,
        "active_hinge_rate": 1.0,
        "amp_overflow_steps": 4,
        "batches": 29,
        "edge_repeat_rate": 0.0,
        "epoch": 1,
        "finite": true,
        "fixed_diagnostic_positive_cosine": 0.8490192890167236,
        "gradient_finite_rate": 0.8620689655172413,
        "hybrid_hard_negative_cosine": 0.7896497845649719,
        "images_per_sec": 46.07015380827805,
        "lambda_over_total": 0.0033257198680577606,
        "lambda_relation_loss": 0.02946995340030769,
        "matched_random_negative_cosine": 0.8114428520202637,
        "max_edge_repeats": 1,
        "negative_edge_coverage": 1.0,
        "nonfinite_gradient_steps": 4,
        "normalized_active_weight_mean": 1.0,
        "normalized_weight_mean": 1.0,
        "normalized_weight_std": 0.0,
        "positive_cosine": 0.8295068227011582,
        "positive_edge_coverage": 1.0,
        "positive_negative_margin": 0.005300476665383783,
        "raw_relation_loss_before_weight": 0.2946995383706586,
        "raw_weight_mean": 1.0,
        "raw_weight_std": 0.0,
        "raw_weight_zero_rate": 0.0,
        "relation_diagnostic_anchor_count": 64,
        "relation_grad_over_total_grad": 0.0026298557792580803,
        "relation_gradient_norm": 0.49153958823474364,
        "sampled_negative_cosine": 0.8242063419572239,
        "sec_per_iter": 1.389185724587276,
        "seconds": 40.286386013031006,
        "semantic_hard_negative_cosine": 0.7896497845649719,
        "successful_optimizer_steps": 25,
        "total_loss": 8.876435575814083,
        "unique_anchors": 1856,
        "unique_negative_images": 987,
        "unique_positive_images": 1798,
        "unique_relation_edges": 1856,
        "visual_backbone_gradient_norm": 186.8542990848936,
        "visual_hard_negative_cosine": 0.8256844878196716,
        "weighted_relation_loss_before_lambda": 0.29469953425999346
      }
    ],
    "run_id": "R04",
    "seed": 0,
    "unexpected_trainable_relation_parameters": 0
  },
  {
    "amp_overflow_steps": 4,
    "finite": true,
    "gradient_finite_on_successful_updates": true,
    "gradient_finite_rate": 0.8571428571428571,
    "iterations": 28,
    "nonfinite_gradient_steps": 4,
    "optimizer_update_success": true,
    "path": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/sanity/R07",
    "phase": "sanity",
    "rows": [
      {
        "H_hybrid": 0.4952713836516653,
        "H_sem": 0.37928806245326996,
        "H_visual": 0.3830474615097046,
        "R_agr": 0.866768981729235,
        "R_conf": 0.6922626921108791,
        "R_joint": 0.6051083207130432,
        "active_edge_rate": 1.0,
        "active_hinge_rate": 1.0,
        "amp_overflow_steps": 4,
        "batches": 28,
        "edge_repeat_rate": 0.0,
        "epoch": 1,
        "finite": true,
        "fixed_diagnostic_positive_cosine": 0.8489285707473755,
        "gradient_finite_rate": 0.8571428571428571,
        "hybrid_hard_negative_cosine": 0.7895449995994568,
        "images_per_sec": 44.873808534220665,
        "lambda_over_total": 0.0033205929544887374,
        "lambda_relation_loss": 0.02942316054499575,
        "matched_random_negative_cosine": 0.8114384412765503,
        "max_edge_repeats": 1,
        "negative_edge_coverage": 1.0,
        "nonfinite_gradient_steps": 4,
        "normalized_active_weight_mean": 1.000000000548815,
        "normalized_weight_mean": 1.000000000548815,
        "normalized_weight_std": 0.30606338151098095,
        "positive_cosine": 0.8294513864176614,
        "positive_edge_coverage": 1.0,
        "positive_negative_margin": 0.005338391056284308,
        "raw_relation_loss_before_weight": 0.2946616251553808,
        "raw_weight_mean": 0.6051083197983514,
        "raw_weight_std": 0.18568742504175112,
        "raw_weight_zero_rate": 0.0,
        "relation_diagnostic_anchor_count": 64,
        "relation_grad_over_total_grad": 0.002804849180995919,
        "relation_gradient_norm": 0.5268181384045596,
        "sampled_negative_cosine": 0.8241129964590073,
        "sec_per_iter": 1.4262217112949915,
        "seconds": 39.934207916259766,
        "semantic_hard_negative_cosine": 0.7895449995994568,
        "successful_optimizer_steps": 24,
        "total_loss": 8.877123355865479,
        "unique_anchors": 1792,
        "unique_negative_images": 962,
        "unique_positive_images": 1739,
        "unique_relation_edges": 1792,
        "visual_backbone_gradient_norm": 187.77152797154017,
        "visual_hard_negative_cosine": 0.8256332874298096,
        "weighted_relation_loss_before_lambda": 0.29423160425254274
      }
    ],
    "run_id": "R07",
    "seed": 0,
    "unexpected_trainable_relation_parameters": 0
  },
  {
    "amp_overflow_steps": 4,
    "finite": true,
    "gradient_finite_on_successful_updates": true,
    "gradient_finite_rate": 0.8620689655172413,
    "iterations": 29,
    "nonfinite_gradient_steps": 4,
    "optimizer_update_success": true,
    "path": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/sanity/R11",
    "phase": "sanity",
    "rows": [
      {
        "H_hybrid": 0.49838362377265405,
        "H_sem": 0.38089836465901344,
        "H_visual": 0.3828833935589626,
        "R_agr": 0.8671875082213303,
        "R_conf": 0.6919865341022097,
        "R_joint": 0.6050584275147011,
        "active_edge_rate": 0.9477370689655172,
        "active_hinge_rate": 1.0,
        "amp_overflow_steps": 4,
        "batches": 29,
        "edge_repeat_rate": 0.0,
        "epoch": 1,
        "finite": true,
        "fixed_diagnostic_positive_cosine": 0.8490332961082458,
        "gradient_finite_rate": 0.8620689655172413,
        "hybrid_hard_negative_cosine": 0.7896372675895691,
        "images_per_sec": 45.88786525014682,
        "lambda_over_total": 0.0031348161142447897,
        "lambda_relation_loss": 0.02775609718057616,
        "matched_random_negative_cosine": 0.8114397525787354,
        "max_edge_repeats": 1,
        "negative_edge_coverage": 1.0,
        "nonfinite_gradient_steps": 4,
        "normalized_active_weight_mean": 0.9999999998983434,
        "normalized_weight_mean": 0.9477370688691735,
        "normalized_weight_std": 0.6742714747251658,
        "positive_cosine": 0.8295107360543876,
        "positive_edge_coverage": 1.0,
        "positive_negative_margin": 0.005301900818173228,
        "raw_relation_loss_before_weight": 0.29469811402518176,
        "raw_weight_mean": 0.30257190013021346,
        "raw_weight_std": 0.2161772345810349,
        "raw_weight_zero_rate": 0.052262931034482756,
        "relation_diagnostic_anchor_count": 64,
        "relation_grad_over_total_grad": 0.0031273354989866517,
        "relation_gradient_norm": 0.5873917953627548,
        "sampled_negative_cosine": 0.8242088268543112,
        "sec_per_iter": 1.3947042350111336,
        "seconds": 40.446422815322876,
        "semantic_hard_negative_cosine": 0.7896372675895691,
        "successful_optimizer_steps": 25,
        "total_loss": 8.874696336943527,
        "unique_anchors": 1856,
        "unique_negative_images": 987,
        "unique_positive_images": 1798,
        "unique_relation_edges": 1856,
        "visual_backbone_gradient_norm": 187.77213892443427,
        "visual_hard_negative_cosine": 0.8256794214248657,
        "weighted_relation_loss_before_lambda": 0.277560968851221
      }
    ],
    "run_id": "R11",
    "seed": 0,
    "unexpected_trainable_relation_parameters": 0
  }
]
```

sanity 必须走真实 relation tuple、forward、relation loss、backward、AMP scaler、optimizer step；text encoder trainable parameters=0，relation graph 无 trainable graph。

## 10. Runtime benchmark and gate

```json
{
  "queue": {
    "completed_count": 13,
    "elapsed_seconds": 71803.69955921173,
    "experiment": "RCHRL-V1",
    "finished": 1789470973.0524826,
    "max_concurrent_jobs": 2,
    "max_jobs_per_device": 1,
    "next_queue_index": 13,
    "queue_length": 24,
    "remaining_tasks": [
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R13_full_l020",
        "priority": 1,
        "run_id": "R13",
        "seed": 0
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R14_positive_only",
        "priority": 1,
        "run_id": "R14",
        "seed": 0
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R15_negative_only",
        "priority": 1,
        "run_id": "R15",
        "seed": 0
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R00",
        "priority": 2,
        "run_id": "R00",
        "seed": 1
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R04",
        "priority": 2,
        "run_id": "R04",
        "seed": 1
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R07",
        "priority": 2,
        "run_id": "R07",
        "seed": 1
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R11",
        "priority": 2,
        "run_id": "R11",
        "seed": 1
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R00",
        "priority": 3,
        "run_id": "R00",
        "seed": 2
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R04",
        "priority": 3,
        "run_id": "R04",
        "seed": 2
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R07",
        "priority": 3,
        "run_id": "R07",
        "seed": 2
      },
      {
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R11",
        "priority": 3,
        "run_id": "R11",
        "seed": 2
      }
    ],
    "runtime_gate_reached": true,
    "started": 1789399169.3529234,
    "status": "runtime_gate_stop",
    "tasks": [
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29620",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R00",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R00_control",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 3641.753966331482,
        "finished": 1789402811.1640415,
        "gpu": "0",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R00_control",
        "pid": 879333,
        "priority": 1,
        "return_code": 0,
        "run_id": "R00",
        "seed": 0,
        "started": 1789399169.4100752,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29621",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R01",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R01_matched_random",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11389.846681594849,
        "finished": 1789410559.3124142,
        "gpu": "1",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R01_matched_random",
        "pid": 879334,
        "priority": 1,
        "return_code": 0,
        "run_id": "R01",
        "seed": 0,
        "started": 1789399169.4657326,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29622",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R02",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R02_visualhard",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11420.004411697388,
        "finished": 1789414233.2942302,
        "gpu": "0",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R02_visualhard",
        "pid": 951010,
        "priority": 1,
        "return_code": 0,
        "run_id": "R02",
        "seed": 0,
        "started": 1789402813.2898185,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29623",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R03",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R03_semhard",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11299.778899669647,
        "finished": 1789421861.271973,
        "gpu": "1",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R03_semhard",
        "pid": 1095564,
        "priority": 1,
        "return_code": 0,
        "run_id": "R03",
        "seed": 0,
        "started": 1789410561.4930732,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29624",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R04",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R04_hybrid",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11239.76629281044,
        "finished": 1789425475.1870298,
        "gpu": "0",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R04_hybrid",
        "pid": 1164851,
        "priority": 1,
        "return_code": 0,
        "run_id": "R04",
        "seed": 0,
        "started": 1789414235.420737,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29625",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R05",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R05_hybrid_conf",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11259.696380376816,
        "finished": 1789433123.1266842,
        "gpu": "1",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R05_hybrid_conf",
        "pid": 1305711,
        "priority": 1,
        "return_code": 0,
        "run_id": "R05",
        "seed": 0,
        "started": 1789421863.4303038,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29626",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R06",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R06_hybrid_agreement",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11249.709439516068,
        "finished": 1789436727.0836005,
        "gpu": "0",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R06_hybrid_agreement",
        "pid": 1359324,
        "priority": 1,
        "return_code": 0,
        "run_id": "R06",
        "seed": 0,
        "started": 1789425477.374161,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29627",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R07",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R07_hybrid_joint",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11285.811413288116,
        "finished": 1789444411.1079917,
        "gpu": "1",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R07_hybrid_joint",
        "pid": 1469667,
        "priority": 1,
        "return_code": 0,
        "run_id": "R07",
        "seed": 0,
        "started": 1789433125.2965784,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29628",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R08",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R08_visual_joint",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11433.90821814537,
        "finished": 1789448163.1033318,
        "gpu": "0",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R08_visual_joint",
        "pid": 1522434,
        "priority": 1,
        "return_code": 0,
        "run_id": "R08",
        "seed": 0,
        "started": 1789436729.1951137,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29629",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R09",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R09_sem_joint",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11315.799929380417,
        "finished": 1789455729.0235116,
        "gpu": "1",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R09_sem_joint",
        "pid": 1633289,
        "priority": 1,
        "return_code": 0,
        "run_id": "R09",
        "seed": 0,
        "started": 1789444413.2235823,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29630",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R10",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R10_hybrid_hardness",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11379.912613153458,
        "finished": 1789459545.1208572,
        "gpu": "0",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R10_hybrid_hardness",
        "pid": 1702087,
        "priority": 1,
        "return_code": 0,
        "run_id": "R10",
        "seed": 0,
        "started": 1789448165.208244,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29631",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R11",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R11_full",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11315.866885185242,
        "finished": 1789467046.995962,
        "gpu": "1",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R11_full",
        "pid": 1841860,
        "priority": 1,
        "return_code": 0,
        "run_id": "R11",
        "seed": 0,
        "started": 1789455731.1290767,
        "status": "complete"
      },
      {
        "command": [
          "/data/envs/PDF/bin/python",
          "-m",
          "torch.distributed.run",
          "--nproc_per_node=1",
          "--master_port",
          "29632",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/tools/run_rchrl.py",
          "--run-id",
          "R12",
          "--seed",
          "0",
          "--graph",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/relation_graph",
          "--output",
          "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R12_full_l005",
          "--phase",
          "train"
        ],
        "elapsed_seconds": 11425.78149485588,
        "finished": 1789470973.0150137,
        "gpu": "0",
        "output": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed0/R12_full_l005",
        "pid": 1913076,
        "priority": 1,
        "return_code": 0,
        "run_id": "R12",
        "seed": 0,
        "started": 1789459547.2335188,
        "status": "complete"
      }
    ]
  },
  "runtime_gate": {
    "created_utc": "2026-09-14T15:07:50Z",
    "estimated_all_24_two_device_lower_bound_hours": 35.56690649614112,
    "estimated_all_24_two_device_relation_wave_hours": 35.65346082454631,
    "estimated_formal_run_hours": {
      "R00": {
        "estimated_evaluation_seconds": 435.0,
        "estimated_formal_run_hours": 1.0227050181706745,
        "estimated_formal_run_seconds": 3681.7380654144285,
        "estimated_training_seconds": 3246.7380654144285
      },
      "R04": {
        "estimated_evaluation_seconds": 435.0,
        "estimated_formal_run_hours": 3.2412237113223914,
        "estimated_formal_run_seconds": 11668.40536076061,
        "estimated_training_seconds": 11233.40536076061
      }
    },
    "estimated_seed0_two_device_hours": 25.92978969057913,
    "experiment": "RCHRL-V1",
    "gate_decision": "exceeds_18h",
    "gate_time_hours": 18.0,
    "priority_1": "seed0 R00-R15",
    "priority_2": "seed1 R00/R04/R07/R11",
    "priority_3": "seed2 R00/R04/R07/R11",
    "protocol_batches_per_epoch": 271,
    "protocol_epochs": 50,
    "queue_policy": "fixed priority: seed0 R00-R15, then seed1 core, then seed2 core; at the 18h gate stop claiming new tasks and let active tasks finish",
    "smoke_inputs": {
      "R00": {
        "amp_overflow_steps": 18,
        "checkpoint_write": true,
        "duration_seconds": 346.01947355270386,
        "evaluation_feature_shape": [
          3384,
          512
        ],
        "finite": true,
        "gradient_finite_rate": 0.9856,
        "images_per_sec": 231.2006364160356,
        "iterations": 1250,
        "nonfinite_gradient_steps": 18,
        "peak_memory_bytes": 9833007616,
        "phase": "smoke",
        "required_duration_seconds": 300.0,
        "sec_per_iter": 0.23961166534423828,
        "successful_optimizer_steps": 1232
      },
      "R04": {
        "amp_overflow_steps": 9,
        "checkpoint_write": true,
        "duration_seconds": 347.4368598461151,
        "evaluation_feature_shape": [
          3384,
          512
        ],
        "finite": true,
        "gradient_finite_rate": 0.9751381215469613,
        "images_per_sec": 66.68255331844077,
        "iterations": 362,
        "nonfinite_gradient_steps": 9,
        "peak_memory_bytes": 33331377664,
        "phase": "smoke",
        "required_duration_seconds": 300.0,
        "sec_per_iter": 0.8290336059601926,
        "successful_optimizer_steps": 353
      }
    }
  },
  "smoke": [
    {
      "amp_overflow_steps": 18,
      "checkpoint_write": true,
      "duration_seconds": 346.01947355270386,
      "evaluation_feature_shape": [
        3384,
        512
      ],
      "finite": true,
      "gradient_finite_rate": 0.9856,
      "images_per_sec": 231.2006364160356,
      "iterations": 1250,
      "nonfinite_gradient_steps": 18,
      "peak_memory_bytes": 9833007616,
      "phase": "smoke",
      "required_duration_seconds": 300.0,
      "run_id": "R00",
      "sec_per_iter": 0.23961166534423828,
      "successful_optimizer_steps": 1232
    },
    {
      "amp_overflow_steps": 9,
      "checkpoint_write": true,
      "duration_seconds": 347.4368598461151,
      "evaluation_feature_shape": [
        3384,
        512
      ],
      "finite": true,
      "gradient_finite_rate": 0.9751381215469613,
      "images_per_sec": 66.68255331844077,
      "iterations": 362,
      "nonfinite_gradient_steps": 9,
      "peak_memory_bytes": 33331377664,
      "phase": "smoke",
      "required_duration_seconds": 300.0,
      "run_id": "R04",
      "sec_per_iter": 0.8290336059601926,
      "successful_optimizer_steps": 353
    }
  ]
}
```

smoke 覆盖 train dataloader、relation sampler（R04）、forward/backward、AMP step、checkpoint write、image-only evaluation feature extraction/shape。正式调度 max_concurrent_jobs=2、max_jobs_per_device=1。

## 11. 24-run completion status

| Seed | Run | Status | Diff R1 | Diff mAP | Same R1 | Same mAP |
|---:|---|---|---:|---:|---:|---:|
| 0 | R00 | complete | 64.352 | 62.146 | 99.871 | 98.139 |
| 0 | R01 | complete | 64.239 | 62.205 | 99.897 | 98.050 |
| 0 | R02 | complete | 64.719 | 62.305 | 99.897 | 98.094 |
| 0 | R03 | complete | 64.098 | 62.163 | 99.897 | 98.045 |
| 0 | R04 | complete | 64.352 | 62.174 | 99.897 | 98.064 |
| 0 | R05 | complete | 64.296 | 62.248 | 99.897 | 98.075 |
| 0 | R06 | complete | 64.324 | 62.225 | 99.897 | 98.072 |
| 0 | R07 | complete | 64.268 | 62.228 | 99.897 | 98.064 |
| 0 | R08 | complete | 64.663 | 62.350 | 99.897 | 98.099 |
| 0 | R09 | complete | 63.929 | 62.147 | 99.897 | 98.032 |
| 0 | R10 | complete | 64.239 | 62.172 | 99.897 | 98.062 |
| 0 | R11 | complete | 64.239 | 62.228 | 99.897 | 98.078 |
| 0 | R12 | complete | 64.268 | 62.190 | 99.897 | 98.077 |
| 0 | R13 | complete | 64.042 | 62.327 | 99.897 | 98.056 |
| 0 | R14 | complete | 64.776 | 61.871 | 99.897 | 97.979 |
| 0 | R15 | complete | 64.042 | 62.205 | 99.897 | 98.161 |
| 1 | R00 | missing | n/a | n/a | n/a | n/a |
| 1 | R04 | missing | n/a | n/a | n/a | n/a |
| 1 | R07 | missing | n/a | n/a | n/a | n/a |
| 1 | R11 | missing | n/a | n/a | n/a | n/a |
| 2 | R00 | missing | n/a | n/a | n/a | n/a |
| 2 | R04 | missing | n/a | n/a | n/a | n/a |
| 2 | R07 | missing | n/a | n/a | n/a | n/a |
| 2 | R11 | missing | n/a | n/a | n/a | n/a |

## 12. Seed0 full table

| Seed | Run | Status | Diff R1 | Diff mAP | Same R1 | Same mAP |
|---:|---|---|---:|---:|---:|---:|
|  | R00 | complete | 64.352 | 62.146 | 99.871 | 98.139 |
|  | R01 | complete | 64.239 | 62.205 | 99.897 | 98.050 |
|  | R02 | complete | 64.719 | 62.305 | 99.897 | 98.094 |
|  | R03 | complete | 64.098 | 62.163 | 99.897 | 98.045 |
|  | R04 | complete | 64.352 | 62.174 | 99.897 | 98.064 |
|  | R05 | complete | 64.296 | 62.248 | 99.897 | 98.075 |
|  | R06 | complete | 64.324 | 62.225 | 99.897 | 98.072 |
|  | R07 | complete | 64.268 | 62.228 | 99.897 | 98.064 |
|  | R08 | complete | 64.663 | 62.350 | 99.897 | 98.099 |
|  | R09 | complete | 63.929 | 62.147 | 99.897 | 98.032 |
|  | R10 | complete | 64.239 | 62.172 | 99.897 | 98.062 |
|  | R11 | complete | 64.239 | 62.228 | 99.897 | 98.078 |
|  | R12 | complete | 64.268 | 62.190 | 99.897 | 98.077 |
|  | R13 | complete | 64.042 | 62.327 | 99.897 | 98.056 |
|  | R14 | complete | 64.776 | 61.871 | 99.897 | 97.979 |
|  | R15 | complete | 64.042 | 62.205 | 99.897 | 98.161 |

完整 Same/Different Clothes metrics（epoch50 primary）：

| Seed | Run | Status | Diff R1 | Diff R5 | Diff R10 | Diff R20 | Diff mAP | Same R1 | Same R5 | Same R10 | Same R20 | Same mAP |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | R00 | complete | 64.352 | 74.316 | 78.182 | 81.061 | 62.146 | 99.871 | 99.923 | 99.923 | 99.948 | 98.139 |
|  | R01 | complete | 64.239 | 74.146 | 77.787 | 80.779 | 62.205 | 99.897 | 99.923 | 99.923 | 99.923 | 98.050 |
|  | R02 | complete | 64.719 | 74.485 | 78.182 | 80.864 | 62.305 | 99.897 | 99.923 | 99.923 | 99.923 | 98.094 |
|  | R03 | complete | 64.098 | 74.118 | 77.759 | 80.864 | 62.163 | 99.897 | 99.923 | 99.923 | 99.923 | 98.045 |
|  | R04 | complete | 64.352 | 74.231 | 78.041 | 80.864 | 62.174 | 99.897 | 99.923 | 99.923 | 99.923 | 98.064 |
|  | R05 | complete | 64.296 | 74.316 | 77.900 | 80.864 | 62.248 | 99.897 | 99.923 | 99.923 | 99.923 | 98.075 |
|  | R06 | complete | 64.324 | 74.400 | 78.013 | 80.948 | 62.225 | 99.897 | 99.923 | 99.923 | 99.923 | 98.072 |
|  | R07 | complete | 64.268 | 74.457 | 77.985 | 80.948 | 62.228 | 99.897 | 99.923 | 99.923 | 99.923 | 98.064 |
|  | R08 | complete | 64.663 | 74.626 | 77.985 | 80.892 | 62.350 | 99.897 | 99.923 | 99.923 | 99.923 | 98.099 |
|  | R09 | complete | 63.929 | 74.203 | 77.759 | 80.807 | 62.147 | 99.897 | 99.923 | 99.923 | 99.923 | 98.032 |
|  | R10 | complete | 64.239 | 74.231 | 77.928 | 80.807 | 62.172 | 99.897 | 99.923 | 99.923 | 99.923 | 98.062 |
|  | R11 | complete | 64.239 | 74.146 | 77.815 | 80.835 | 62.228 | 99.897 | 99.923 | 99.923 | 99.923 | 98.078 |
|  | R12 | complete | 64.268 | 74.316 | 78.069 | 80.864 | 62.190 | 99.897 | 99.923 | 99.923 | 99.923 | 98.077 |
|  | R13 | complete | 64.042 | 74.033 | 77.844 | 80.977 | 62.327 | 99.897 | 99.923 | 99.923 | 99.923 | 98.056 |
|  | R14 | complete | 64.776 | 74.965 | 77.844 | 81.089 | 61.871 | 99.897 | 99.923 | 99.923 | 99.948 | 97.979 |
|  | R15 | complete | 64.042 | 73.666 | 77.505 | 80.469 | 62.205 | 99.897 | 99.923 | 99.923 | 99.923 | 98.161 |

## 13. Main chain R00 → R04 → R07 → R11

| Transition | Diff R1 delta mean ± std (pp) | Diff mAP delta mean ± std (pp) | positive seeds |
|---|---:|---:|---:|
| R00->R04 | 0.000 ± 0.000 | 0.028 ± 0.000 | R1 0/1; mAP 1/1 |
| R04->R07 | -0.085 ± 0.000 | 0.054 ± 0.000 | R1 0/1; mAP 1/1 |
| R07->R11 | -0.028 ± 0.000 | -0.000 ± 0.000 | R1 0/1; mAP 0/1 |

## 14. Three-seed matched results

| Seed | Run | Status | Diff R1 | Diff mAP | Same R1 | Same mAP |
|---:|---|---|---:|---:|---:|---:|
| 1 | R00 | missing | n/a | n/a | n/a | n/a |
| 1 | R04 | missing | n/a | n/a | n/a | n/a |
| 1 | R07 | missing | n/a | n/a | n/a | n/a |
| 1 | R11 | missing | n/a | n/a | n/a | n/a |
| 2 | R00 | missing | n/a | n/a | n/a | n/a |
| 2 | R04 | missing | n/a | n/a | n/a | n/a |
| 2 | R07 | missing | n/a | n/a | n/a | n/a |
| 2 | R11 | missing | n/a | n/a | n/a | n/a |

| Variant vs same-seed R00 | Metric | seed deltas (pp) | mean (pp) | std (pp) | >0 seeds |
|---|---|---|---:|---:|---:|
| R04 | Diff R1 | 0.000 | 0.000 | 0.000 | 0 |
| R04 | Diff mAP | 0.028 | 0.028 | 0.000 | 1 |
| R04 | Same R1 | 0.026 | 0.026 | 0.000 | 1 |
| R04 | Same mAP | -0.075 | -0.075 | 0.000 | 0 |
| R07 | Diff R1 | -0.085 | -0.085 | 0.000 | 0 |
| R07 | Diff mAP | 0.082 | 0.082 | 0.000 | 1 |
| R07 | Same R1 | 0.026 | 0.026 | 0.000 | 1 |
| R07 | Same mAP | -0.075 | -0.075 | 0.000 | 0 |
| R11 | Diff R1 | -0.113 | -0.113 | 0.000 | 0 |
| R11 | Diff mAP | 0.082 | 0.082 | 0.000 | 1 |
| R11 | Same R1 | 0.026 | 0.026 | 0.000 | 1 |
| R11 | Same mAP | -0.061 | -0.061 | 0.000 | 0 |

## 15. Relation diagnostics

### R00

```json
{
  "H_hybrid": "0.0",
  "H_sem": "0.0",
  "H_visual": "0.0",
  "R_agr": "0.0",
  "R_conf": "0.0",
  "R_joint": "0.0",
  "active_edge_rate": "0.0",
  "active_hinge_rate": "0.0",
  "amp_overflow_steps": "3",
  "batches": "271",
  "edge_repeat_rate": "1.0",
  "epoch": "50",
  "finite": "True",
  "fixed_diagnostic_positive_cosine": "",
  "gradient_finite_rate": "0.988929889298893",
  "hybrid_hard_negative_cosine": "",
  "images_per_sec": "272.91225526132325",
  "lambda_over_total": "0.0",
  "lambda_relation_loss": "0.0",
  "matched_random_negative_cosine": "",
  "max_edge_repeats": "0",
  "negative_edge_coverage": "0.0",
  "nonfinite_gradient_steps": "3",
  "normalized_active_weight_mean": "0.0",
  "normalized_weight_mean": "0.0",
  "normalized_weight_std": "0.0",
  "positive_cosine": "0.0",
  "positive_edge_coverage": "0.0",
  "positive_negative_margin": "0.0",
  "raw_relation_loss_before_weight": "0.0",
  "raw_weight_mean": "0.0",
  "raw_weight_std": "0.0",
  "raw_weight_zero_rate": "0.0",
  "relation_diagnostic_anchor_count": "0",
  "relation_grad_over_total_grad": "0.0",
  "relation_gradient_norm": "0.0",
  "sampled_negative_cosine": "0.0",
  "sec_per_iter": "0.23450760735796827",
  "seconds": "63.5515615940094",
  "semantic_hard_negative_cosine": "",
  "successful_optimizer_steps": "268",
  "total_loss": "5.062738332361313",
  "unique_anchors": "0",
  "unique_negative_images": "0",
  "unique_positive_images": "0",
  "unique_relation_edges": "0",
  "visual_backbone_gradient_norm": "26.252930642933862",
  "visual_hard_negative_cosine": "",
  "weighted_relation_loss_before_lambda": "0.0"
}
```

### R04

```json
{
  "H_hybrid": "0.5004733933953781",
  "H_sem": "0.3825803556895344",
  "H_visual": "0.3816092929277033",
  "R_agr": "0.8681196242680849",
  "R_conf": "0.696067253601947",
  "R_joint": "0.6097067904648306",
  "active_edge_rate": "1.0",
  "active_hinge_rate": "0.7164437269372693",
  "amp_overflow_steps": "3",
  "batches": "271",
  "edge_repeat_rate": "0.0",
  "epoch": "50",
  "finite": "True",
  "fixed_diagnostic_positive_cosine": "0.8446142673492432",
  "gradient_finite_rate": "0.988929889298893",
  "hybrid_hard_negative_cosine": "0.7608143091201782",
  "images_per_sec": "80.13880473825161",
  "lambda_over_total": "0.002669004576514403",
  "lambda_relation_loss": "0.013535783760167137",
  "matched_random_negative_cosine": "0.5272430777549744",
  "max_edge_repeats": "1",
  "negative_edge_coverage": "1.0",
  "nonfinite_gradient_steps": "3",
  "normalized_active_weight_mean": "1.0",
  "normalized_weight_mean": "1.0",
  "normalized_weight_std": "0.0",
  "positive_cosine": "0.860190377684097",
  "positive_edge_coverage": "1.0",
  "positive_negative_margin": "0.19108141485835353",
  "raw_relation_loss_before_weight": "0.1353578383852195",
  "raw_weight_mean": "1.0",
  "raw_weight_std": "0.0",
  "raw_weight_zero_rate": "0.0",
  "relation_diagnostic_anchor_count": "64",
  "relation_grad_over_total_grad": "0.015817865989531162",
  "relation_gradient_norm": "0.424751675526693",
  "sampled_negative_cosine": "0.6691089606812959",
  "sec_per_iter": "0.7986143567905215",
  "seconds": "216.42449069023132",
  "semantic_hard_negative_cosine": "0.7608143091201782",
  "successful_optimizer_steps": "268",
  "total_loss": "5.0719683355071",
  "unique_anchors": "17344",
  "unique_negative_images": "4066",
  "unique_positive_images": "12583",
  "unique_relation_edges": "17344",
  "visual_backbone_gradient_norm": "25.756562393089943",
  "visual_hard_negative_cosine": "0.8149095773696899",
  "weighted_relation_loss_before_lambda": "0.1353578377528824"
}
```

### R07

```json
{
  "H_hybrid": "0.5004733933953781",
  "H_sem": "0.3825803556895344",
  "H_visual": "0.3816092929277033",
  "R_agr": "0.8681196242680849",
  "R_conf": "0.696067253601947",
  "R_joint": "0.6097067904648306",
  "active_edge_rate": "0.999365774907749",
  "active_hinge_rate": "0.7151176199261993",
  "amp_overflow_steps": "2",
  "batches": "271",
  "edge_repeat_rate": "0.0",
  "epoch": "50",
  "finite": "True",
  "fixed_diagnostic_positive_cosine": "0.8434286713600159",
  "gradient_finite_rate": "0.992619926199262",
  "hybrid_hard_negative_cosine": "0.7593081593513489",
  "images_per_sec": "80.01558084642015",
  "lambda_over_total": "0.0026301531214782507",
  "lambda_relation_loss": "0.013338028198980978",
  "matched_random_negative_cosine": "0.5240465998649597",
  "max_edge_repeats": "1",
  "negative_edge_coverage": "1.0",
  "nonfinite_gradient_steps": "2",
  "normalized_active_weight_mean": "0.9999999998461139",
  "normalized_weight_mean": "0.9993657747539606",
  "normalized_weight_std": "0.3015885264902898",
  "positive_cosine": "0.8598839002781689",
  "positive_edge_coverage": "1.0",
  "positive_negative_margin": "0.19170401589017072",
  "raw_relation_loss_before_weight": "0.13512647644070241",
  "raw_weight_mean": "0.609706792700778",
  "raw_weight_std": "0.18458142921296156",
  "raw_weight_zero_rate": "0.0006342250922509225",
  "relation_diagnostic_anchor_count": "64",
  "relation_grad_over_total_grad": "0.016144557524588814",
  "relation_gradient_norm": "0.43620991339529286",
  "sampled_negative_cosine": "0.6681798874672049",
  "sec_per_iter": "0.7998442218752365",
  "seconds": "216.7577841281891",
  "semantic_hard_negative_cosine": "0.7593081593513489",
  "successful_optimizer_steps": "269",
  "total_loss": "5.07180062431251",
  "unique_anchors": "17344",
  "unique_negative_images": "4066",
  "unique_positive_images": "12583",
  "unique_relation_edges": "17344",
  "visual_backbone_gradient_norm": "25.917550745045567",
  "visual_hard_negative_cosine": "0.8135799765586853",
  "weighted_relation_loss_before_lambda": "0.133380281996683"
}
```

### R11

```json
{
  "H_hybrid": "0.5004733933953781",
  "H_sem": "0.3825803556895344",
  "H_visual": "0.3816092929277033",
  "R_agr": "0.8681196242680849",
  "R_conf": "0.696067253601947",
  "R_joint": "0.6097067904648306",
  "active_edge_rate": "0.9499538745387454",
  "active_hinge_rate": "0.7182887453874539",
  "amp_overflow_steps": "3",
  "batches": "271",
  "edge_repeat_rate": "0.0",
  "epoch": "50",
  "finite": "True",
  "fixed_diagnostic_positive_cosine": "0.8450425267219543",
  "gradient_finite_rate": "0.988929889298893",
  "hybrid_hard_negative_cosine": "0.7572463750839233",
  "images_per_sec": "80.1957611702341",
  "lambda_over_total": "0.002468292921011738",
  "lambda_relation_loss": "0.01251636567522129",
  "matched_random_negative_cosine": "0.5292325019836426",
  "max_edge_repeats": "1",
  "negative_edge_coverage": "1.0",
  "nonfinite_gradient_steps": "3",
  "normalized_active_weight_mean": "1.0000000003645928",
  "normalized_weight_mean": "0.9499538748850918",
  "normalized_weight_std": "0.6634694735328334",
  "positive_cosine": "0.861228664422827",
  "positive_edge_coverage": "1.0",
  "positive_negative_margin": "0.1897213887904403",
  "raw_relation_loss_before_weight": "0.13602941502504243",
  "raw_weight_mean": "0.30511381036487156",
  "raw_weight_std": "0.21375882316052505",
  "raw_weight_zero_rate": "0.050046125461254615",
  "relation_diagnostic_anchor_count": "64",
  "relation_grad_over_total_grad": "0.017542922017592465",
  "relation_gradient_norm": "0.47355335065069654",
  "sampled_negative_cosine": "0.67150727387284",
  "sec_per_iter": "0.7980471669087991",
  "seconds": "216.27078223228455",
  "semantic_hard_negative_cosine": "0.7572463750839233",
  "successful_optimizer_steps": "268",
  "total_loss": "5.071399159097144",
  "unique_anchors": "17344",
  "unique_negative_images": "4066",
  "unique_positive_images": "12583",
  "unique_relation_edges": "17344",
  "visual_backbone_gradient_norm": "25.90018507299388",
  "visual_hard_negative_cosine": "0.8161233067512512",
  "weighted_relation_loss_before_lambda": "0.12516365667660737"
}
```

## 16. Semantic-confuser before/after analysis

```json
{
  "feature_definition": "raw projected CLS encode_image, L2 normalized; no flip",
  "fixed_anchor_count": 100,
  "hybrid_top1": {
    "count": 400,
    "mean_delta": -0.06965050905942917,
    "mean_rank_after": 236.4125,
    "mean_rank_before": 233.68,
    "mean_similarity_after": 0.7909498426318169,
    "mean_similarity_before": 0.860600351691246,
    "rank_improved_fraction": 0.6425
  },
  "missing": [
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R00/epoch50_final.pth",
      "run_id": "R00",
      "seed": 1
    },
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R04/epoch50_final.pth",
      "run_id": "R04",
      "seed": 1
    },
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R07/epoch50_final.pth",
      "run_id": "R07",
      "seed": 1
    },
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed1/R11/epoch50_final.pth",
      "run_id": "R11",
      "seed": 1
    },
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R00/epoch50_final.pth",
      "run_id": "R00",
      "seed": 2
    },
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R04/epoch50_final.pth",
      "run_id": "R04",
      "seed": 2
    },
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R07/epoch50_final.pth",
      "run_id": "R07",
      "seed": 2
    },
    {
      "checkpoint": "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/crossclothes_relation_v1/seed2/R11/epoch50_final.pth",
      "run_id": "R11",
      "seed": 2
    }
  ],
  "rows": 800,
  "semantic_confusers": "frozen graph semantic_top1 and hybrid_top1 different-ID train negatives",
  "semantic_top1": {
    "count": 400,
    "mean_delta": -0.06965050905942917,
    "mean_rank_after": 236.4125,
    "mean_rank_before": 233.68,
    "mean_similarity_after": 0.7909498426318169,
    "mean_similarity_before": 0.860600351691246,
    "rank_improved_fraction": 0.6425
  },
  "test_data_used": false
}
```

固定 train semantic-top1/hybrid-top1 confusers 的 before/after raw projected CLS similarity 与 visual rank 由独立 post-training train-only analysis 生成；TEST 不参与。

## 17. Same/Different Clothes image-only metrics

主表已分别列出 Same R1/mAP 与 Diff R1/mAP；每个 run 的 `eval_epoch50.json`/`evaluation_history.json` 还包含 R5/R10/R20。primary 为 epoch50 final，best-test epoch 仅 auxiliary。

## 18. Failure cases and validation boundary

```json
{
  "checkpoint_selection_for_mining": "not test-adaptive",
  "failures": [
    "8 formal runs missing/incomplete"
  ],
  "relation_graph_test_data": false,
  "test_labels_scope": "final image-only evaluation metrics only",
  "validation_status": "Needs revision"
}
```

## 19. Final scientific conclusion

**E — 所有 matched gains 仍接近 seed variance。在当前 frozen P2 semantic signal + PDF backbone + relation formulation 下，semantic reliability 与 semantic-confusion-aware hard-negative mining 不足以产生稳定提升。**

边界：即使 E，也不能外推为“semantic hard-negative mining 在 CC-ReID 中无效”；只能作当前 frozen P2 signal、PDF backbone 与 relation formulation 下的结论。

## 20. Next-step recommendation

若主链没有超过 seed variance，下一步停止增加文本复杂度，转向纯视觉 cross-clothing relation learning；若 R04 稳定而 R07/R11 不稳定，则保留 hybrid selection 但移除未证实的 weighting。

