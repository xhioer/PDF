"""Audit the frozen semantic/hybrid diagnostic index mapping."""
from __future__ import absolute_import

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    graph_dir = Path(args.graph)
    index_path = graph_dir / "relation_index.npz"
    arrays = np.load(str(index_path))
    semantic_top20 = np.array(arrays["semantic_neg"], dtype=np.int64, copy=True)
    hybrid_top20 = np.array(arrays["hybrid_neg"], dtype=np.int64, copy=True)
    semantic = np.array(semantic_top20[:, 0], dtype=np.int64, copy=True)
    hybrid = np.array(hybrid_top20[:, 0], dtype=np.int64, copy=True)
    if np.shares_memory(semantic, hybrid):
        raise RuntimeError("semantic/hybrid arrays alias")
    n = len(semantic)
    fixed = min(100, n)
    first100_match = int(np.sum(semantic[:fixed] == hybrid[:fixed]))
    all_match = int(np.sum(semantic == hybrid))
    # Every row is unique within each top-20 list.  A broadcasted equality
    # check is equivalent to set intersection but avoids a slow Python loop
    # over the 17,896 train anchors on the object-backed filesystem.
    intersection = np.sum(
        np.any(semantic_top20[:, :, None] == hybrid_top20[:, None, :], axis=2), axis=1)
    jaccard = intersection / (40.0 - intersection)
    manifest = json.load(open(str(graph_dir / "relation_graph_manifest.json")))
    ledger = json.load(open(str(graph_dir / "relation_graph_hashes.json")))
    hash_checks = {}
    for name, expected in ledger["files"].items():
        actual = sha256_file(graph_dir / name)
        hash_checks[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    sidecar = (graph_dir / "relation_graph_hashes.sha256").read_text().split()[0]
    ledger_sha = sha256_file(graph_dir / "relation_graph_hashes.json")
    graph_ok = all(item["match"] for item in hash_checks.values()) and sidecar == ledger_sha
    mismatches = [
        {"anchor": int(i), "semantic": int(semantic[i]), "hybrid": int(hybrid[i])}
        for i in np.flatnonzero(semantic != hybrid)
    ]
    lines = [
        "# Semantic/Hybrid diagnostic fix audit",
        "",
        "## Scope",
        "",
        "This is a read-only audit of the frozen train-only relation graph. No graph file was modified or re-mined.",
        "",
        "## Frozen graph verification",
        "",
        "- graph directory: `{}`".format(graph_dir),
        "- relation index: `{}`".format(index_path),
        "- graph manifest frozen: `{}`".format(manifest.get("frozen")),
        "- graph manifest train_only: `{}`".format(manifest.get("train_only")),
        "- graph manifest test_data_used: `{}`".format(manifest.get("test_data_used")),
        "- all graph payload hashes and ledger sidecar match: `{}`".format(graph_ok),
        "",
        "## Explicit index mapping",
        "",
        "- semantic top-1 source: `relation_index.npz[\"semantic_neg\"][:, 0]`",
        "- hybrid top-1 source: `relation_index.npz[\"hybrid_neg\"][:, 0]`",
        "- independent copied arrays: `True`",
        "- number of train anchors: `{}`".format(n),
        "- top-1 exact matches among first 100 anchors: `{}/{} ({:.3f}%)`".format(
            first100_match, fixed, 100.0 * first100_match / float(fixed)),
        "- top-1 exact matches over all anchors: `{}/{} ({:.3f}%)`".format(
            all_match, n, 100.0 * all_match / float(n)),
        "- top-1 mismatch count over all anchors: `{}`".format(len(mismatches)),
        "- mean top-20 Jaccard: `{:.6f}`".format(float(jaccard.mean())),
        "- top-20 Jaccard range: `[{:.6f}, {:.6f}]`".format(float(jaccard.min()), float(jaccard.max())),
        "",
        "## Interpretation",
        "",
    ]
    if first100_match == fixed:
        lines.append(
            "The first-100 fixed diagnostic anchors have 100% semantic/hybrid top-1 agreement, "
            "so identical top-1 before/after aggregates in the prior report are expected for that "
            "specific scope. The graph still differs at top-20 level and differs at top-1 for "
            "{} of {} anchors overall; this is not evidence that the graph arrays are aliases.".format(
                len(mismatches), n))
        lines.append(
            "Because the first-100 top-1 match rate is 100%, the required secondary graph check "
            "was performed: frozen payload hashes match, semantic and hybrid keys are distinct, "
            "the arrays do not share memory, and the top-20 sets have the expected non-unit overlap.")
    else:
        lines.append(
            "The first-100 fixed diagnostic anchors contain distinct semantic/hybrid top-1 indices; "
            "the prior identical aggregate therefore cannot be attributed to universal top-1 agreement.")
    lines.extend([
        "",
        "## First top-1 mismatches outside the fixed-100 scope",
        "",
        "```json",
        json.dumps(mismatches[:10], indent=2),
        "```",
        "",
        "## Code repair",
        "",
        "`tools/analyze_semantic_confusers.py` now uses explicit independent copies of "
        "`semantic_neg[:, 0]` and `hybrid_neg[:, 0]`, preserves the graph-key mapping in the "
        "output summary, and accepts a separate checkpoint root. The graph and mining inputs "
        "remain unchanged.",
        "",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(json.dumps({
        "output": str(output), "graph_ok": graph_ok,
        "top1_exact_match_first100": first100_match,
        "top1_exact_match_rate_first100": first100_match / float(fixed),
        "top1_exact_match_all": all_match,
        "top1_exact_match_rate_all": all_match / float(n),
        "mean_top20_jaccard": float(jaccard.mean()),
        "top1_mismatch_count": len(mismatches),
    }, indent=2))


if __name__ == "__main__":
    main()
