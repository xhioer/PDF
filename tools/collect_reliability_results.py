"""Collect only complete, protocol-matched final-epoch V0–V4 runs."""
import argparse
import csv
import json
from pathlib import Path
import yaml


def main(root, reports):
    variants = [('V0', 'V0_original', 'Original PDF', 'No', 'No'),
                ('V1', 'V1_p2_none', 'P2', 'No', 'No'),
                ('V2', 'V2_p2_attribute', 'P2', 'Yes', 'No'),
                ('V3', 'V3_p2_confidence', 'P2', 'No', 'Yes'),
                ('V4', 'V4_p2_joint', 'P2', 'Yes', 'Yes')]
    rows, protocols, commits, inputs = [], [], [], []
    for variant, directory, semantic, attribute, confidence in variants:
        run = root / directory
        status = json.loads((run / 'status.json').read_text())
        manifest = json.loads((run / 'run_manifest.json').read_text())
        if status['status'] != 'complete' or status['phase'] != 'train' or status['final_epoch'] != 50:
            raise ValueError('Run is not a completed epoch50 formal training: ' + directory)
        if not manifest['tracked_diff_empty']:
            raise ValueError('Dirty training source: ' + directory)
        commits.append(manifest['git_commit'])
        inputs.append((manifest['pretrained_sha256'], manifest['caption_sha256']))
        config = yaml.safe_load(manifest['config'])
        config.pop('SEMANTIC_RELIABILITY_MODE')
        config['DATA'].pop('SEMANTIC_CACHE')
        protocols.append(json.dumps(config, sort_keys=True))
        results = json.loads((run / 'evaluation.json').read_text())
        final = next(result for result in results if result['epoch'] == 50)
        rows.append(dict(Variant=variant, Semantic=semantic, AttributeReliability=attribute,
            ConfidenceReliability=confidence, Diff_R1=100 * final['different']['Rank-1'],
            Diff_mAP=100 * final['different']['mAP'], Same_R1=100 * final['same']['Rank-1'],
            Same_mAP=100 * final['same']['mAP']))
    if len(set(protocols)) != 1 or len(set(commits)) != 1 or len(set(inputs)) != 1:
        raise ValueError('Protocol, commit or source inputs differ; refusing comparison')
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / 'pdf_reliability_ablation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = []
    by_variant = {row['Variant']: row for row in rows}
    for a, b in [('V1', 'V0'), ('V2', 'V1'), ('V3', 'V1'), ('V4', 'V1'), ('V4', 'V2'), ('V4', 'V3')]:
        comparisons.append({'Comparison': a + '-' + b, **{metric: by_variant[a][metric] - by_variant[b][metric]
            for metric in ('Diff_R1', 'Diff_mAP', 'Same_R1', 'Same_mAP')}})
    with (reports / 'pdf_reliability_deltas.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print('Wrote verified final-epoch comparison; deltas are percentage points.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/pdf_reliability_ablation'))
    parser.add_argument('--reports', type=Path, default=Path('reports'))
    args = parser.parse_args()
    main(args.root, args.reports)
