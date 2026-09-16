# Semantic/Hybrid diagnostic fix audit

## Scope

This is a read-only audit of the frozen train-only relation graph. No graph file was modified or re-mined.

## Frozen graph verification

- graph directory: `outputs/crossclothes_relation_confirm/relation_graph`
- relation index: `outputs/crossclothes_relation_confirm/relation_graph/relation_index.npz`
- graph manifest frozen: `True`
- graph manifest train_only: `True`
- graph manifest test_data_used: `False`
- all graph payload hashes and ledger sidecar match: `True`

## Explicit index mapping

- semantic top-1 source: `relation_index.npz["semantic_neg"][:, 0]`
- hybrid top-1 source: `relation_index.npz["hybrid_neg"][:, 0]`
- independent copied arrays: `True`
- number of train anchors: `17896`
- top-1 exact matches among first 100 anchors: `100/100 (100.000%)`
- top-1 exact matches over all anchors: `17499/17896 (97.782%)`
- top-1 mismatch count over all anchors: `397`
- mean top-20 Jaccard: `0.334874`
- top-20 Jaccard range: `[0.333333, 0.428571]`

## Interpretation

The first-100 fixed diagnostic anchors have 100% semantic/hybrid top-1 agreement, so identical top-1 before/after aggregates in the prior report are expected for that specific scope. The graph still differs at top-20 level and differs at top-1 for 397 of 17896 anchors overall; this is not evidence that the graph arrays are aliases.
Because the first-100 top-1 match rate is 100%, the required secondary graph check was performed: frozen payload hashes match, semantic and hybrid keys are distinct, the arrays do not share memory, and the top-20 sets have the expected non-unit overlap.

## First top-1 mismatches outside the fixed-100 scope

```json
[
  {
    "anchor": 293,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 297,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 302,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 304,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 306,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 331,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 351,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 356,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 357,
    "semantic": 11733,
    "hybrid": 16145
  },
  {
    "anchor": 445,
    "semantic": 11247,
    "hybrid": 7079
  }
]
```

## Code repair

`tools/analyze_semantic_confusers.py` now uses explicit independent copies of `semantic_neg[:, 0]` and `hybrid_neg[:, 0]`, preserves the graph-key mapping in the output summary, and accepts a separate checkpoint root. The graph and mining inputs remain unchanged.
