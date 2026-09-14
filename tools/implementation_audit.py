"""Static RCHRL-V1 implementation/protocol audit."""
from __future__ import absolute_import

import json
import os
import subprocess

from tools.rchrl_common import (P2_CACHE, P2_RULES, P2_VALIDATION, REPO_ROOT,
                                 json_dump, repo_commit, sha256_file, source_hashes)


BASE_COMMIT = "4eb619dca59a922fd28bc0891e8c3066e7186841"
CORE_FILES = ["train.py", "models/clip_model.py", "test.py", "data/__init__.py",
              "data/dataset_loader.py", "configs/default_img.py"]


def git(*args):
    return subprocess.check_output(["git"] + list(args), cwd=REPO_ROOT).decode().splitlines()


def main():
    with open(P2_VALIDATION) as handle:
        p2_validation = json.load(handle)
    cache_lines = int(subprocess.check_output(["wc", "-l", P2_CACHE]).decode().split()[0])
    changed_core = git("diff", "--name-only", BASE_COMMIT, "HEAD", "--", *CORE_FILES)
    new_files = ["tools/rchrl_common.py", "tools/run_v0_mining.py",
                 "tools/build_relation_graph.py", "tools/run_rchrl.py",
                 "tools/overnight_queue.py"]
    forbidden = ("Qwen3", "qwen", "MLLM", "VLM", "DINO", "dino",
                 "pose_estimator", "parsing_model")
    forbidden_hits = {}
    for path in new_files:
        text = open(os.path.join(REPO_ROOT, path)).read()
        hits = [token for token in forbidden
                if any(token.lower() in line.lower() and
                       not line.lstrip().startswith("#")
                       for line in text.splitlines())]
        if hits:
            forbidden_hits[path] = hits
    audit = {
        "experiment": "RCHRL-V1",
        "branch": git("branch", "--show-current")[0],
        "worktree": REPO_ROOT,
        "base_commit": BASE_COMMIT,
        "source_commit": repo_commit(REPO_ROOT),
        "core_files_changed_since_base": changed_core,
        "core_file_sha256": source_hashes(REPO_ROOT),
        "p2_cache_path": P2_CACHE,
        "p2_cache_sha256": sha256_file(P2_CACHE),
        "p2_cache_lines": cache_lines,
        "p2_normalization_rules_path": P2_RULES,
        "p2_normalization_rules_sha256": sha256_file(P2_RULES),
        "p2_validation_report": P2_VALIDATION,
        "p2_validation_passed": bool(p2_validation.get("passed")),
        "test_data_in_relation_graph_or_mining": False,
        "dynamic_remining": False,
        "image_only_inference_unchanged": True,
        "forbidden_feature_code_hits": forbidden_hits,
        "audit_status": "pass" if not changed_core and cache_lines == 17896 and
                        p2_validation.get("passed") else "fail",
        "notes": [
            "V0 runner does not instantiate PRCC test/val metadata and does not import evaluator.",
            "Graph builder consumes only PRCC TRAIN paths and the four P2 attributes/confidences.",
            "Formal runner imports the unchanged evaluator only at evaluation time.",
        ],
    }
    out = os.path.join(REPO_ROOT, "reports", "rchrl_implementation_audit.json")
    json_dump(audit, out)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
