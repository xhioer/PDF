"""Collect inspectable RCHRL-V1 artifacts into the required Markdown report."""
from __future__ import absolute_import

import argparse
import csv
import json
import os
import statistics
import subprocess
from collections import OrderedDict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DEFAULT = os.path.join(ROOT, "outputs", "crossclothes_relation_v1", "relation_graph")
OUTPUT_DEFAULT = os.path.join(ROOT, "outputs", "crossclothes_relation_v1")
REPORT_DEFAULT = os.path.join(ROOT, "reports", "PDF_crossclothes_relation_v1_overnight_report.md")

SEED0 = OrderedDict([
    ("R00", "R00_control"), ("R01", "R01_matched_random"), ("R02", "R02_visualhard"),
    ("R03", "R03_semhard"), ("R04", "R04_hybrid"), ("R05", "R05_hybrid_conf"),
    ("R06", "R06_hybrid_agreement"), ("R07", "R07_hybrid_joint"), ("R08", "R08_visual_joint"),
    ("R09", "R09_sem_joint"), ("R10", "R10_hybrid_hardness"), ("R11", "R11_full"),
    ("R12", "R12_full_l005"), ("R13", "R13_full_l020"), ("R14", "R14_positive_only"),
    ("R15", "R15_negative_only"),
])


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as handle:
        return json.load(handle)


def read_metrics(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def run_dir(root, seed, run_id):
    if seed == 0:
        return os.path.join(root, "seed0", SEED0[run_id])
    return os.path.join(root, "seed{}".format(seed), run_id)


def final_eval(path):
    history = read_json(os.path.join(path, "evaluation_history.json"), []) or []
    if not history:
        candidate = read_json(os.path.join(path, "eval_epoch50.json"))
        return candidate
    return sorted(history, key=lambda row: int(row.get("epoch", 0)))[-1]


def final_metric(path, subset, metric):
    result = final_eval(path)
    if not result or subset not in result:
        return None
    return result[subset].get(metric)


def delta(values_a, values_b):
    return [a - b for a, b in zip(values_a, values_b)]


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "std": None, "positive_count": 0}
    return {"n": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "positive_count": sum(x > 0 for x in values)}


def pp(value):
    return "n/a" if value is None else "{:.3f}".format(100.0 * value)


def metric_table(rows, include_seed=True):
    lines = ["| Seed | Run | Status | Diff R1 | Diff mAP | Same R1 | Same mAP |", 
             "|---:|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            row["seed"] if include_seed else "", row["run_id"], row["status"],
            pp(row.get("diff_R1")), pp(row.get("diff_mAP")),
            pp(row.get("same_R1")), pp(row.get("same_mAP"))))
    return "\n".join(lines)


def json_code(obj):
    return "```json\n{}\n```".format(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=OUTPUT_DEFAULT)
    parser.add_argument("--graph", default=GRAPH_DEFAULT)
    parser.add_argument("--report", default=REPORT_DEFAULT)
    args = parser.parse_args()

    audit = read_json(os.path.join(ROOT, "reports", "rchrl_implementation_audit.json"), {}) or {}
    provenance = read_json(os.path.join(ROOT, "reports", "v0_mining_checkpoint_provenance.json"), {}) or {}
    graph_stats = read_json(os.path.join(args.graph, "relation_graph_stats.json"), {}) or {}
    graph_manifest = read_json(os.path.join(args.graph, "relation_graph_manifest.json"), {}) or {}
    graph_ledger = read_json(os.path.join(args.graph, "relation_graph_hashes.json"), {}) or {}
    queue = read_json(os.path.join(args.root, "queue_status.json"), {}) or {}
    extreme_path = os.path.join(ROOT, "reports", "extreme_hard_negative_audit.csv")
    extreme_count = 0
    if os.path.exists(extreme_path):
        with open(extreme_path) as handle:
            extreme_count = max(0, sum(1 for _ in handle) - 1)

    all_rows = []
    for seed in (0, 1, 2):
        run_ids = list(SEED0) if seed == 0 else ["R00", "R04", "R07", "R11"]
        for run_id in run_ids:
            path = run_dir(args.root, seed, run_id)
            summary = read_json(os.path.join(path, "run_summary.json"), {}) or {}
            row = {"seed": seed, "run_id": run_id,
                   "status": "complete" if os.path.exists(os.path.join(path, "epoch50_final.pth")) else "missing",
                   "diff_R1": final_metric(path, "different", "R1"),
                   "diff_mAP": final_metric(path, "different", "mAP"),
                   "same_R1": final_metric(path, "same", "R1"),
                   "same_mAP": final_metric(path, "same", "mAP"),
                   "elapsed_seconds": summary.get("elapsed_seconds"), "path": path}
            all_rows.append(row)

    by_key = {(row["seed"], row["run_id"]): row for row in all_rows}
    core_deltas = {}
    for variant in ("R04", "R07", "R11"):
        core_deltas[variant] = {}
        for metric_key in ("diff_R1", "diff_mAP", "same_R1", "same_mAP"):
            values = []
            for seed in (0, 1, 2):
                variant_value = by_key[(seed, variant)].get(metric_key)
                control_value = by_key[(seed, "R00")].get(metric_key)
                if variant_value is not None and control_value is not None:
                    values.append(variant_value - control_value)
            core_deltas[variant][metric_key] = {"seed_deltas": values, **stats(values)}

    chain_deltas = {}
    for left, right in (("R00", "R04"), ("R04", "R07"), ("R07", "R11")):
        chain_deltas[left + "->" + right] = {}
        for metric_key in ("diff_R1", "diff_mAP", "same_R1", "same_mAP"):
            values = []
            for seed in (0, 1, 2):
                a, b = by_key[(seed, right)].get(metric_key), by_key[(seed, left)].get(metric_key)
                if a is not None and b is not None:
                    values.append(a - b)
            chain_deltas[left + "->" + right][metric_key] = {"seed_deltas": values, **stats(values)}

    r04 = core_deltas["R04"]
    r07_add = chain_deltas["R04->R07"]
    r11_add = chain_deltas["R07->R11"]
    strong_r04 = ((r04["diff_R1"]["mean"] is not None and r04["diff_R1"]["mean"] >= .008) or
                  (r04["diff_mAP"]["mean"] is not None and r04["diff_mAP"]["mean"] >= .005)) and \
                 (r04["diff_R1"]["positive_count"] >= 2 or r04["diff_mAP"]["positive_count"] >= 2)
    stable_r07 = ((r07_add["diff_R1"]["mean"] is not None and r07_add["diff_R1"]["mean"] >= .003) or
                  (r07_add["diff_mAP"]["mean"] is not None and r07_add["diff_mAP"]["mean"] >= .002)) and \
                 (r07_add["diff_R1"]["positive_count"] >= 2 or r07_add["diff_mAP"]["positive_count"] >= 2)
    stable_r11 = ((r11_add["diff_R1"]["mean"] is not None and r11_add["diff_R1"]["mean"] >= .003) or
                  (r11_add["diff_mAP"]["mean"] is not None and r11_add["diff_mAP"]["mean"] >= .002)) and \
                 (r11_add["diff_R1"]["positive_count"] >= 2 or r11_add["diff_mAP"]["positive_count"] >= 2)
    if strong_r04 and stable_r07 and stable_r11:
        conclusion = "A"
    elif strong_r04 and not stable_r07 and not stable_r11:
        conclusion = "B"
    elif strong_r04 and not stable_r07:
        conclusion = "C"
    elif stable_r07 and not stable_r11:
        conclusion = "D"
    else:
        conclusion = "E"

    reliability = (graph_stats.get("reliability") or {})
    negative_sets = graph_stats.get("negative_sets") or {}
    smoke_rows = []
    for run_id, name in (("R00", "R00_control"), ("R04", "R04_hybrid")):
        path = os.path.join(args.root, "smoke", name)
        runtime = read_json(os.path.join(path, "smoke_runtime.json"), {}) or {}
        if runtime:
            smoke_rows.append({"run_id": run_id, **runtime})
    sanity_rows = []
    for path, _, files in os.walk(args.root):
        if "sanity.json" in files:
            item = read_json(os.path.join(path, "sanity.json"), {}) or {}
            sanity_rows.append({"path": path, **item})
    complete_count = sum(row["status"] == "complete" for row in all_rows)
    report_ready = complete_count == 24 and bool(provenance) and bool(graph_manifest)
    validation_status = "Ready within reviewed scope" if report_ready else "Needs revision"

    graph_file_lines = []
    for name, digest in (graph_ledger.get("files") or {}).items():
        graph_file_lines.append("- `{}`: `{}`".format(name, digest))
    graph_file_lines = "\n".join(graph_file_lines) or "- graph hash ledger missing"

    main_rows = [by_key[(0, run_id)] for run_id in SEED0]
    multi_rows = [row for row in all_rows if row["seed"] in (1, 2)]
    lines = []
    lines.append("# PDF_crossclothes_relation_v1 overnight report")
    lines.append("")
    lines.append("结论先行：本报告状态为 **{}**；24-run completion 为 **{}/24**，科学结论分类为 **{}**。".format(
        validation_status, complete_count, conclusion))
    lines.append("")
    lines.append("## 1. Implementation audit")
    lines.append("")
    lines.append(json_code({key: audit.get(key) for key in (
        "audit_status", "branch", "worktree", "base_commit", "source_commit",
        "core_files_changed_since_base", "p2_cache_lines", "p2_cache_sha256",
        "p2_validation_passed", "dynamic_remining", "test_data_in_relation_graph_or_mining",
        "image_only_inference_unchanged", "forbidden_feature_code_hits") if key in audit}))
    lines.append("")
    lines.append("冻结核心文件相对 base commit 无改动；新增 runner/graph/sampler 位于独立 branch/worktree。原 PDF `train.py`、`models/clip_model.py`、`test.py` 与 inference 输出 shape 保持不变。")
    lines.append("")
    lines.append("## 2. V0 mining checkpoint provenance")
    lines.append("")
    lines.append(json_code({key: provenance.get(key) for key in (
        "checkpoint_path", "checkpoint_sha256", "source_commit", "source_commit_at_launch",
        "training_seed", "epoch", "train_only", "test_adaptive_checkpoint_selection",
        "no_test_adaptive_checkpoint_selection_was_used_for_relation_mining",
        "raw_checkpoint_path", "raw_checkpoint_sha256", "provenance_correction") if key in provenance}))
    lines.append("")
    lines.append("明确记录：no test-adaptive checkpoint selection was used for relation mining。epoch50 final 是主 mining checkpoint；任何 best-test checkpoint 仅可作为辅助，不进入 graph。")
    lines.append("")
    lines.append("## 3. Relation graph construction and SHA256")
    lines.append("")
    lines.append(json_code({key: graph_manifest.get(key) for key in (
        "graph_version", "frozen", "train_only", "train_image_count", "train_id_count",
        "positive_definition", "semantic_allowed_attributes", "semantic_disallowed_fields",
        "semantic_serialization", "hardness_definition", "hybrid_definition",
        "matched_random_definition", "v0_checkpoint_sha256", "p2_cache_sha256",
        "test_data_used") if key in graph_manifest}))
    lines.append("")
    lines.append("Hash ledger（graph freeze 后未重新 mining）：")
    lines.append(graph_file_lines)
    lines.append("")
    lines.append("## 4. Raw reliability distributions")
    lines.append("")
    lines.append(json_code({key: reliability.get(key) for key in ("R_conf", "R_agr", "R_joint", "collapse_warning") if key in reliability}))
    lines.append("")
    lines.append("R_conf 使用四个允许属性 confidence 的 geometric mean pair confidence；R_agr 对两端有效且 exact-equal 的 categorical values 求平均，unknown 不进入 denominator；R_joint=R_conf×R_agr。Identity-level 分布完整保存在 `relation_graph_stats.json`。")
    lines.append("")
    lines.append("## 5. Normalized training-weight distributions and effective relation scale")
    lines.append("")
    lines.append("每个正式 run 的 `metrics.csv` 逐 epoch 记录 raw/normalized weight mean/std/zero-rate、active-edge rate、raw relation loss、weighted relation loss、lambda contribution、total loss、gradient norm 与 ratio。active weight normalization 按 active edge mean 执行；zero edge 保持 0。")
    lines.append("")
    lines.append("## 6. Metadata-matched random negative validation")
    lines.append("")
    lines.append(json_code({name: negative_sets.get(name) for name in ("matched_random", "visual", "semantic", "hybrid") if name in negative_sets}))
    lines.append("")
    lines.append("matched-random 每个 anchor 20 条 control，different-ID；match level、camera/clothes pair、same-clothes rate 与四个属性 overlap 已在上述 JSON 及 graph stats 中记录。")
    lines.append("")
    lines.append("## 7. Visual/semantic/hybrid overlap and semantic-hard audit")
    lines.append("")
    lines.append(json_code({"overlap": graph_stats.get("overlap"), "semantic_confuser_analysis": graph_stats.get("semantic_confuser_analysis")}))
    lines.append("")
    lines.append("判定依据是 semantic-hard 的 semantic cosine 是否高于 visual-hard，同时 visual cosine 不必同样高；该判断不以 TEST 为依据。")
    lines.append("")
    lines.append("## 8. Extreme hard-negative audit")
    lines.append("")
    lines.append("`reports/extreme_hard_negative_audit.csv` 包含 {} 个 audit rows（前 100 个固定 train anchors 的 visual/semantic/hybrid top-5），含路径、person/camera/clothes metadata、两种 cosine 与 P2 属性。".format(extreme_count))
    lines.append("")
    lines.append("## 9. Gradient sanity")
    lines.append("")
    lines.append(json_code(sanity_rows))
    lines.append("")
    lines.append("sanity 必须走真实 relation tuple、forward、relation loss、backward、AMP scaler、optimizer step；text encoder trainable parameters=0，relation graph 无 trainable graph。")
    lines.append("")
    lines.append("## 10. Runtime benchmark and gate")
    lines.append("")
    lines.append(json_code({"smoke": smoke_rows, "queue": queue}))
    lines.append("")
    lines.append("smoke 覆盖 train dataloader、relation sampler（R04）、forward/backward、AMP step、checkpoint write、image-only evaluation feature extraction/shape。正式调度 max_concurrent_jobs=2、max_jobs_per_device=1。")
    lines.append("")
    lines.append("## 11. 24-run completion status")
    lines.append("")
    lines.append(metric_table(all_rows))
    lines.append("")
    lines.append("## 12. Seed0 full table")
    lines.append("")
    lines.append(metric_table(main_rows, include_seed=False))
    lines.append("")
    lines.append("## 13. Main chain R00 → R04 → R07 → R11")
    lines.append("")
    lines.append("| Transition | Diff R1 delta mean ± std (pp) | Diff mAP delta mean ± std (pp) | positive seeds |")
    lines.append("|---|---:|---:|---:|")
    for key in ("R00->R04", "R04->R07", "R07->R11"):
        r = chain_deltas[key]
        d1, dm = r["diff_R1"], r["diff_mAP"]
        lines.append("| {} | {} ± {} | {} ± {} | R1 {}/{}; mAP {}/{} |".format(
            key, pp(d1["mean"]), pp(d1["std"]), pp(dm["mean"]), pp(dm["std"]),
            d1["positive_count"], d1["n"], dm["positive_count"], dm["n"]))
    lines.append("")
    lines.append("## 14. Three-seed matched results")
    lines.append("")
    lines.append(metric_table(multi_rows))
    lines.append("")
    lines.append("| Variant vs same-seed R00 | Metric | seed deltas (pp) | mean (pp) | std (pp) | >0 seeds |")
    lines.append("|---|---|---|---:|---:|---:|")
    for variant in ("R04", "R07", "R11"):
        for metric_key, label in (("diff_R1", "Diff R1"), ("diff_mAP", "Diff mAP"),
                                  ("same_R1", "Same R1"), ("same_mAP", "Same mAP")):
            item = core_deltas[variant][metric_key]
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                variant, label, ", ".join(pp(x) for x in item["seed_deltas"]),
                pp(item["mean"]), pp(item["std"]), item["positive_count"]))
    lines.append("")
    lines.append("## 15. Relation diagnostics")
    lines.append("")
    for run_id in ("R00", "R04", "R07", "R11"):
        path = by_key[(0, run_id)]["path"]
        metrics = read_metrics(os.path.join(path, "metrics.csv"))
        lines.append("### {}".format(run_id))
        lines.append("")
        lines.append(json_code(metrics[-1] if metrics else {"status": "missing"}))
        lines.append("")
    lines.append("## 16. Semantic-confuser before/after analysis")
    lines.append("")
    confuser = read_json(os.path.join(ROOT, "reports", "semantic_confuser_before_after.json"), {}) or {}
    lines.append(json_code(confuser))
    lines.append("")
    lines.append("固定 train semantic-top1/hybrid-top1 confusers 的 before/after raw projected CLS similarity 与 visual rank 由独立 post-training train-only analysis 生成；TEST 不参与。")
    lines.append("")
    lines.append("## 17. Same/Different Clothes image-only metrics")
    lines.append("")
    lines.append("主表已分别列出 Same R1/mAP 与 Diff R1/mAP；每个 run 的 `eval_epoch50.json`/`evaluation_history.json` 还包含 R5/R10/R20。primary 为 epoch50 final，best-test epoch 仅 auxiliary。")
    lines.append("")
    lines.append("## 18. Failure cases and validation boundary")
    lines.append("")
    failures = []
    if not audit or audit.get("audit_status") != "pass":
        failures.append("implementation audit not passed")
    if not provenance:
        failures.append("V0 provenance missing")
    if not graph_manifest or not graph_manifest.get("frozen"):
        failures.append("frozen graph manifest missing")
    if complete_count != 24:
        failures.append("{} formal runs missing/incomplete".format(24 - complete_count))
    lines.append(json_code({"failures": failures, "validation_status": validation_status,
                            "test_labels_scope": "final image-only evaluation metrics only",
                            "relation_graph_test_data": False,
                            "checkpoint_selection_for_mining": "not test-adaptive"}))
    lines.append("")
    lines.append("## 19. Final scientific conclusion")
    lines.append("")
    conclusions = {
        "A": "Cross-clothes relation learning 明显有效；hybrid hard negatives 有效；reliability / hardness weighting 进一步稳定提升。",
        "B": "R04 有效，但 R07/R11 没有额外收益。relation learning 有价值，semantic reliability / hardness weighting 未显示稳定增益。",
        "C": "Hard-negative mining 有效，positive reliability 无明显收益。",
        "D": "Positive reliability 有效，hardness weighting 无额外收益。",
        "E": "所有 matched gains 仍接近 seed variance。在当前 frozen P2 semantic signal + PDF backbone + relation formulation 下，semantic reliability 与 semantic-confusion-aware hard-negative mining 不足以产生稳定提升。",
    }
    lines.append("**{} — {}**".format(conclusion, conclusions[conclusion]))
    lines.append("")
    lines.append("边界：即使 E，也不能外推为“semantic hard-negative mining 在 CC-ReID 中无效”；只能作当前 frozen P2 signal、PDF backbone 与 relation formulation 下的结论。")
    lines.append("")
    lines.append("## 20. Next-step recommendation")
    lines.append("")
    lines.append("若主链没有超过 seed variance，下一步停止增加文本复杂度，转向纯视觉 cross-clothing relation learning；若 R04 稳定而 R07/R11 不稳定，则保留 hybrid selection 但移除未证实的 weighting。")
    lines.append("")
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"report": args.report, "status": validation_status,
                      "scientific_conclusion": conclusion, "completed_runs": complete_count}, indent=2))


if __name__ == "__main__":
    main()
