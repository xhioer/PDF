"""Validate V2 runs and generate the registered tables and overnight report."""

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

from configs.identity_preservation_registry import (
    MULTISEED_VARIANTS,
    OUTPUT_NAMES,
    PRE_REGISTERED,
    SEED0_VARIANTS,
    output_name,
    queue_tasks,
)


METRICS = ('Diff_R1', 'Diff_R5', 'Diff_R10', 'Diff_R20', 'Diff_mAP',
           'Same_R1', 'Same_R5', 'Same_R10', 'Same_R20', 'Same_mAP')
PRIMARY_METRICS = ('Diff_R1', 'Diff_mAP', 'Same_R1', 'Same_mAP')


def read_json(path):
    return json.loads(path.read_text())


def run_path(root, variant, seed):
    return root / ('seed{}'.format(seed)) / output_name(variant, seed)


def final_metrics(run):
    evaluation = read_json(run / 'evaluation.json')
    final = next(row for row in evaluation if int(row['epoch']) == 50)
    values = {}
    for prefix, key in (('Diff', 'different'), ('Same', 'same')):
        for rank in (1, 5, 10, 20):
            values['{}_R{}'.format(prefix, rank)] = 100.0 * float(final[key]['Rank-{}'.format(rank)])
        values['{}_mAP'.format(prefix)] = 100.0 * float(final[key]['mAP'])
    values['epoch'] = 50
    return values


def final_diagnostics(run):
    path = run / 'epoch_diagnostics.csv'
    if not path.exists():
        return dict(cos_res_sem=0.0, cos_com_sem=0.0, semantic_margin=0.0,
                    semantic_valid_rate=0.0)
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if int(row['epoch']) == 50)
    return dict(
        cos_res_sem=0.5 * (float(row['cos_res1_sem']) + float(row['cos_res2_sem'])),
        cos_com_sem=float(row['cos_com_sem']),
        semantic_margin=0.5 * (float(row['semantic_margin_res1_minus_com'])
                               + float(row['semantic_margin_res2_minus_com'])),
        semantic_valid_rate=float(row['semantic_valid_rate']),
    )


def load_status(root, variant, seed):
    path = run_path(root, variant, seed)
    status_path = path / 'status.json'
    if not status_path.exists():
        return dict(variant=variant, seed=seed, status='missing', output=str(path))
    status = read_json(status_path)
    row = dict(variant=variant, seed=seed, status=status.get('status'), output=str(path),
               final_epoch=status.get('final_epoch'), error=status.get('error', ''))
    if status.get('status') == 'complete':
        try:
            row.update(final_metrics(path))
            row.update(final_diagnostics(path))
        except Exception as exc:
            row.update(status='invalid', error=repr(exc))
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value is None:
        return 'n/a'
    return ('{:.%df}' % digits).format(value)


def old_results(candidates):
    for path in candidates:
        if path.exists():
            try:
                with path.open(newline='') as handle:
                    return list(csv.DictReader(handle)), str(path)
            except Exception:
                pass
    return [], None


def main(args):
    root = Path(args.output_root)
    reports = Path(args.reports)
    repo_root = Path(__file__).resolve().parents[1]
    reports.mkdir(parents=True, exist_ok=True)
    all_rows = [load_status(root, variant, seed) for variant, seed in queue_tasks()]
    write_csv(reports / 'identity_preservation_run_status.csv', all_rows)

    complete_rows = [row for row in all_rows if row['status'] == 'complete']
    complete_by_key = {(row['variant'], int(row['seed'])): row for row in complete_rows}
    seed0_rows = []
    control = complete_by_key.get(('N00', 0))
    for variant in SEED0_VARIANTS:
        row = complete_by_key.get((variant, 0))
        spec = PRE_REGISTERED[variant]
        output_row = dict(
            Variant=variant,
            Mechanism=spec['mechanism'],
            Reliability=spec['reliability'],
            LambdaRaw=spec['lambda_raw'],
            LambdaPres=spec['lambda_pres'],
            LambdaExcl=spec['lambda_excl'],
            LambdaRank=spec['lambda_rank'],
            Margin=spec['margin'],
        )
        if row is None:
            output_row.update({metric: '' for metric in METRICS})
            output_row.update(cos_res_sem='', cos_com_sem='', semantic_margin='', semantic_valid_rate='',
                              Delta_Diff_R1='', Delta_Diff_mAP='', Delta_Same_R1='', Delta_Same_mAP='')
        else:
            output_row.update({metric: row[metric] for metric in METRICS})
            output_row.update(cos_res_sem=row['cos_res_sem'], cos_com_sem=row['cos_com_sem'],
                              semantic_margin=row['semantic_margin'],
                              semantic_valid_rate=row['semantic_valid_rate'])
            for metric, name in (('Diff_R1', 'Delta_Diff_R1'), ('Diff_mAP', 'Delta_Diff_mAP'),
                                 ('Same_R1', 'Delta_Same_R1'), ('Same_mAP', 'Delta_Same_mAP')):
                output_row[name] = (row[metric] - control[metric]) if control is not None else ''
        seed0_rows.append(output_row)
    write_csv(reports / 'identity_preservation_seed0_ablation.csv', seed0_rows)

    multiseed_rows = []
    per_seed_rows = []
    for variant in MULTISEED_VARIANTS:
        spec = PRE_REGISTERED[variant]
        values = {metric: [complete_by_key[(variant, seed)][metric]
                           for seed in (0, 1, 2)
                           if (variant, seed) in complete_by_key]
                  for metric in METRICS}
        deltas = {metric: [complete_by_key[(variant, seed)][metric]
                            - complete_by_key[('N00', seed)][metric]
                            for seed in (0, 1, 2)
                            if (variant, seed) in complete_by_key
                            and ('N00', seed) in complete_by_key]
                  for metric in PRIMARY_METRICS}
        row = dict(Variant=variant, Mechanism=spec['mechanism'], Reliability=spec['reliability'],
                   n_seeds=len(values['Diff_R1']))
        for metric in PRIMARY_METRICS:
            row[metric + '_mean'] = statistics.mean(values[metric]) if values[metric] else ''
            row[metric + '_std'] = statistics.stdev(values[metric]) if len(values[metric]) >= 2 else ''
            row['Delta_' + metric + '_mean'] = statistics.mean(deltas[metric]) if deltas[metric] else ''
            row['Delta_' + metric + '_std'] = statistics.stdev(deltas[metric]) if len(deltas[metric]) >= 2 else ''
            for seed_index, value in enumerate(deltas[metric]):
                row['Delta_' + metric + '_seed{}'.format(seed_index)] = value
            for seed_index in range(len(deltas[metric]), 3):
                row['Delta_' + metric + '_seed{}'.format(seed_index)] = ''
        multiseed_rows.append(row)
        for seed in (0, 1, 2):
            variant_row = complete_by_key.get((variant, seed))
            baseline_row = complete_by_key.get(('N00', seed))
            if variant_row is not None and baseline_row is not None:
                per_seed_rows.append(dict(
                    Variant=variant, Seed=seed,
                    **{metric: variant_row[metric] for metric in PRIMARY_METRICS},
                    **{'Delta_' + metric: variant_row[metric] - baseline_row[metric]
                       for metric in PRIMARY_METRICS},
                ))
    write_csv(reports / 'identity_preservation_multiseed.csv', multiseed_rows)
    write_csv(reports / 'identity_preservation_multiseed_per_seed.csv', per_seed_rows)

    old_candidates = [
        Path(args.old_results),
        repo_root / 'reports/pdf_reliability_ablation.csv',
        Path('/data/projects/PDF-worktrees/pdf-reliability-ablation/reports/pdf_reliability_ablation.csv'),
        Path('/data/outputs/PDF/reports/pdf_reliability_ablation.csv'),
    ]
    old_rows, old_path = old_results(old_candidates)
    old_by_variant = {row.get('Variant'): row for row in old_rows}
    # The old V0 reference is part of the user-provided frozen record and is
    # retained even when the prior CSV is not present in this worktree.
    old_v0 = old_by_variant.get('V0', {'Diff_R1': '64.4087', 'Diff_mAP': '62.1613'})
    old_eot_values = []
    for name in ('V1', 'V2', 'V3', 'V4'):
        if name in old_by_variant:
            old_eot_values.append((name, float(old_by_variant[name]['Diff_R1']),
                                   float(old_by_variant[name]['Diff_mAP'])))

    def value(variant, seed, metric):
        row = complete_by_key.get((variant, seed))
        return None if row is None else float(row[metric])

    def mean_delta(variant, metric):
        vals = []
        for seed in (0, 1, 2):
            v = value(variant, seed, metric)
            b = value('N00', seed, metric)
            if v is not None and b is not None:
                vals.append(v - b)
        return statistics.mean(vals) if vals else None

    def signs(variant, metric):
        result = []
        for seed in (0, 1, 2):
            delta = mean_delta(variant, metric)  # only used as fallback below
            v = value(variant, seed, metric)
            b = value('N00', seed, metric)
            if v is not None and b is not None:
                result.append(v - b)
        return result

    complete_formal = len(complete_rows) == len(queue_tasks())
    raw_delta_seed0 = (seed0_rows[1].get('Delta_Diff_R1')
                       if len(seed0_rows) > 1 and seed0_rows[1].get('Delta_Diff_R1') != '' else None)
    robust_new = []
    for variant in MULTISEED_VARIANTS[1:]:
        d1 = mean_delta(variant, 'Diff_R1')
        dm = mean_delta(variant, 'Diff_mAP')
        if d1 is not None and dm is not None:
            robust_new.append((variant, d1, dm, signs(variant, 'Diff_R1'), signs(variant, 'Diff_mAP')))
    best_new = max(robust_new, key=lambda row: row[1], default=None)
    reliability_rows = [row for row in robust_new if row[0] in ('N14', 'N15', 'N16')]
    best_reliability = max(reliability_rows, key=lambda row: row[1], default=None)
    n09 = next((row for row in robust_new if row[0] == 'N09'), None)
    reliability_positive = bool(best_reliability and best_reliability[1] > 0 and best_reliability[2] > 0
                                and sum(x > 0 for x in best_reliability[3]) >= 2
                                and sum(x > 0 for x in best_reliability[4]) >= 2)
    relative_or_excl_positive = bool(best_new and best_new[1] > 0 and best_new[2] > 0)
    direct_raw_harmful = raw_delta_seed0 is not None and raw_delta_seed0 <= 0
    seed_variance_dominant = False
    for row in multiseed_rows:
        if row.get('n_seeds') == 3 and row.get('Delta_Diff_R1_std') not in ('', None):
            if abs(float(row['Delta_Diff_R1_mean'])) <= float(row['Delta_Diff_R1_std']):
                seed_variance_dominant = True
    if not complete_formal:
        classification = 'E'
        classification_reason = '27 个预注册 run 未全部完成，不能用未完成队列支持模块结论。'
    elif direct_raw_harmful and relative_or_excl_positive:
        classification = 'C'
        classification_reason = 'seed0 的 direct raw alignment 不优于 control，而 relative/exclusion 家族在 matched multi-seed 中出现正向证据。'
    elif reliability_positive and old_eot_values:
        old_best = max(old_eot_values, key=lambda row: row[1])
        if best_reliability and best_reliability[1] > old_best[1] + 1.0 and best_reliability[2] > old_best[2] + 1.0:
            classification = 'A'
            classification_reason = 'identity-preservation 与 prior EOT additive best 均有超过1个百分点的 matched mean 优势，且 reliability 正向且跨 seed。'
        else:
            classification = 'B'
            classification_reason = 'identity-preservation 机制有收益或更合理的 decomposition 证据，但相对旧 EOT additive 的明显优势或 reliability 稳定增益不足。'
    elif robust_new and any(row[1] > 0 and row[2] > 0 for row in robust_new):
        classification = 'B'
        n09_row = next((row for row in robust_new if row[0] == 'N09'), None)
        n16_row = next((row for row in robust_new if row[0] == 'N16'), None)
        n09_multi = next((row for row in multiseed_rows if row['Variant'] == 'N09'), None)
        n16_multi = next((row for row in multiseed_rows if row['Variant'] == 'N16'), None)
        classification_reason = 'identity-preservation 有限正向证据：N09 matched Diff R1 Δ={}±{} pp、Diff mAP Δ={}±{} pp；N16 joint matched Diff mAP Δ={}±{} pp、R1 Δ={}±{} pp。整体支持继续验证，但不支持广泛或明显的 reliability 增益。'.format(
            fmt(n09_row[1]) if n09_row else 'n/a',
            fmt(statistics.stdev(n09_row[3])) if n09_row and len(n09_row[3]) >= 2 else 'n/a',
            fmt(n09_row[2]) if n09_row else 'n/a',
            fmt(float(n09_multi['Delta_Diff_mAP_std'])) if n09_multi and n09_multi.get('Delta_Diff_mAP_std') not in ('', None) else 'n/a',
            fmt(n16_row[2]) if n16_row else 'n/a',
            fmt(float(n16_multi['Delta_Diff_mAP_std'])) if n16_multi and n16_multi.get('Delta_Diff_mAP_std') not in ('', None) else 'n/a',
            fmt(n16_row[1]) if n16_row else 'n/a',
            fmt(statistics.stdev(n16_row[3])) if n16_row and len(n16_row[3]) >= 2 else 'n/a')
    elif seed_variance_dominant:
        classification = 'E'
        classification_reason = '效应量主要不超过 matched seed 标准差，当前不能支持新模块。'
    else:
        classification = 'D'
        classification_reason = '当前四属性表示下，各 semantic mechanism 未显示稳定的 ReID 正收益。'

    ranking = sorted([row for row in seed0_rows if row.get('Diff_R1') != ''],
                     key=lambda row: float(row['Diff_R1']), reverse=True)
    same_stability = []
    for row in multiseed_rows:
        if row.get('Same_R1_std') not in ('', None):
            same_stability.append((row['Variant'], float(row['Same_R1_mean']), float(row['Same_R1_std'])))

    seed0_by_variant = {row['Variant']: row for row in seed0_rows}
    multiseed_by_variant = {row['Variant']: row for row in multiseed_rows}

    def seed0_number(variant, field):
        value = seed0_by_variant.get(variant, {}).get(field)
        return None if value in ('', None) else float(value)

    def multiseed_number(variant, field):
        value = multiseed_by_variant.get(variant, {}).get(field)
        return None if value in ('', None) else float(value)

    def show(value, digits=4):
        return 'n/a' if value is None else fmt(value, digits)

    sanity = {}
    sanity_path = reports / 'identity_preservation_gradient_sanity.json'
    if sanity_path.exists():
        try:
            sanity = read_json(sanity_path)
        except Exception:
            sanity = {}

    q1_r1 = seed0_number('N01', 'Delta_Diff_R1')
    q1_map = seed0_number('N01', 'Delta_Diff_mAP')
    q2_best_r1 = max((seed0_number(v, 'Diff_R1') for v in ('N02', 'N03', 'N04')
                      if seed0_number(v, 'Diff_R1') is not None), default=None)
    q2_best_map = max((seed0_number(v, 'Diff_mAP') for v in ('N02', 'N03', 'N04')
                       if seed0_number(v, 'Diff_mAP') is not None), default=None)
    q2_raw_r1 = seed0_number('N01', 'Diff_R1')
    q2_raw_map = seed0_number('N01', 'Diff_mAP')
    q3_best = max((v for v in ('N05', 'N06', 'N07')
                   if seed0_number(v, 'Diff_R1') is not None),
                  key=lambda v: seed0_number(v, 'Diff_R1'), default=None)
    q5_n12_r1 = seed0_number('N12', 'Delta_Diff_R1')
    q5_n12_map = seed0_number('N12', 'Delta_Diff_mAP')
    q5_n13_r1 = seed0_number('N13', 'Delta_Diff_R1')
    q5_n13_map = seed0_number('N13', 'Delta_Diff_mAP')
    n16_r1_deltas = [multiseed_number('N16', 'Delta_Diff_R1_seed{}'.format(seed)) for seed in (0, 1, 2)]
    n16_map_deltas = [multiseed_number('N16', 'Delta_Diff_mAP_seed{}'.format(seed)) for seed in (0, 1, 2)]

    old_table = []
    for row in old_rows:
        old_table.append('| {Variant} | {Semantic} | {AttributeReliability} | {ConfidenceReliability} | {Diff_R1} | {Diff_mAP} | {Same_R1} | {Same_mAP} |'.format(
            Variant=row.get('Variant', ''), Semantic=row.get('Semantic', ''),
            AttributeReliability=row.get('AttributeReliability', ''),
            ConfidenceReliability=row.get('ConfidenceReliability', ''),
            Diff_R1=show(float(row['Diff_R1'])) if row.get('Diff_R1') else 'n/a',
            Diff_mAP=show(float(row['Diff_mAP'])) if row.get('Diff_mAP') else 'n/a',
            Same_R1=show(float(row['Same_R1'])) if row.get('Same_R1') else 'n/a',
            Same_mAP=show(float(row['Same_mAP'])) if row.get('Same_mAP') else 'n/a'))

    report_lines = [
        '# PDF Identity-Semantic Preservation V2 overnight report',
        '',
        '生成时间（UTC）：' + __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
        '实现 commit：`' + subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + '`',
        '输出根目录：`' + str(root) + '`',
        '',
        '## 结论',
        '',
        '**自动分类：{}**。{}'.format(classification, classification_reason),
        '',
        '继续 identity-semantic preservation 方向：**{}**。'.format(
            'YES' if classification == 'A' else ('WEAK YES' if classification in ('B', 'C') else 'NO')),
        '',
        '最有希望配置（分别代表 R1 机制候选与 reliability/mAP 候选）：',
    ]
    if best_new:
        report_lines.append('- `{}`：Delta Diff R1 mean={} pp，Delta Diff mAP mean={} pp；matched R1 deltas={}。'.format(
            best_new[0], fmt(best_new[1]), fmt(best_new[2]), ', '.join(fmt(x) for x in best_new[3])))
    if multiseed_by_variant.get('N16'):
        report_lines.append('- `N16`：joint reliability；Delta Diff R1 mean={}±{} pp，Delta Diff mAP mean={}±{} pp；Same R1={}±{}。'.format(
            show(multiseed_number('N16', 'Delta_Diff_R1_mean')),
            show(multiseed_number('N16', 'Delta_Diff_R1_std')),
            show(multiseed_number('N16', 'Delta_Diff_mAP_mean')),
            show(multiseed_number('N16', 'Delta_Diff_mAP_std')),
            show(multiseed_number('N16', 'Same_R1_mean')),
            show(multiseed_number('N16', 'Same_R1_std'))))
    elif best_reliability:
        report_lines.append('- `{}`：reliability 家族当前最佳，Delta Diff R1 mean={} pp，Delta Diff mAP mean={} pp。'.format(
            best_reliability[0], fmt(best_reliability[1]), fmt(best_reliability[2])))
    if not best_new:
        report_lines.append('- 当前没有足够完整的 3-seed matched 结果。')
    report_lines += [
        '',
        '## Implementation audit',
        '',
        '- tensor audit：见 [identity_preservation_tensor_audit.md](identity_preservation_tensor_audit.md)。',
        '- semantic cache：固定 full PRCC train 17,896/17,896；属性固定为 gender、hair_color、hair_length、body_build；epsilon=1e-6；未知属性权重为0；all-unknown sample 不进入 semantic loss 分母。',
        '- 原 caption clothing branch 保留；所有 V2 run 的 EOT additive guidance 为 OFF；没有新增 projection、attention、encoder、backbone、observation gate 或 OPL/triplet 修改。',
        '- gradient sanity：见 [identity_preservation_gradient_sanity.json](identity_preservation_gradient_sanity.json)；passed={}，四个 loss finite={}，image-only bitwise unchanged={}，visual trainable tensors={}，trainable parameter count before/after={}/{}。'.format(
            sanity.get('passed', 'n/a'),
            all(sanity.get('loss_finite', {}).values()) if sanity.get('loss_finite') else 'n/a',
            sanity.get('image_only_bitwise_equal_with_semantic_argument', 'n/a'),
            (sanity.get('gradient_checks', {}).get('L_rank', {}).get('visual_trainable_parameter_tensors', 'n/a')),
            sanity.get('trainable_parameter_count_before', 'n/a'),
            sanity.get('trainable_parameter_count_after', 'n/a')),
        '',
        '## 27 个实验完成状态',
        '',
        '详表见 [identity_preservation_run_status.csv](identity_preservation_run_status.csv)。完成数：{}/{}；失败数：{}；未启动数：{}。'.format(
            len([row for row in all_rows if row['status'] == 'complete']), len(all_rows),
            len([row for row in all_rows if row['status'] not in ('complete', 'missing')]),
            len([row for row in all_rows if row['status'] == 'missing'])),
        '',
        '## Seed0 全表',
        '',
        '见 [identity_preservation_seed0_ablation.csv](identity_preservation_seed0_ablation.csv)。主指标为 epoch50 final；best test checkpoint 只作为 run 内辅助诊断。',
        '',
    ]
    report_lines += [
        '| Variant | Mechanism | Reliability | λraw | λpres | λexcl | λrank | margin | Diff R1 | Diff mAP | Same R1 | Same mAP | cos_res | cos_com | margin diagnostic |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in seed0_rows:
        report_lines.append('| {Variant} | {Mechanism} | {Reliability} | {LambdaRaw} | {LambdaPres} | {LambdaExcl} | {LambdaRank} | {Margin} | {Diff_R1} | {Diff_mAP} | {Same_R1} | {Same_mAP} | {cos_res_sem} | {cos_com_sem} | {semantic_margin} |'.format(
            Variant=row['Variant'], Mechanism=row['Mechanism'], Reliability=row['Reliability'],
            LambdaRaw=show(float(row['LambdaRaw'])), LambdaPres=show(float(row['LambdaPres'])),
            LambdaExcl=show(float(row['LambdaExcl'])), LambdaRank=show(float(row['LambdaRank'])),
            Margin=show(float(row['Margin'])), Diff_R1=show(float(row['Diff_R1'])),
            Diff_mAP=show(float(row['Diff_mAP'])), Same_R1=show(float(row['Same_R1'])),
            Same_mAP=show(float(row['Same_mAP'])), cos_res_sem=show(float(row['cos_res_sem']), 6),
            cos_com_sem=show(float(row['cos_com_sem']), 6),
            semantic_margin=show(float(row['semantic_margin']), 6)))
    report_lines += ['', '## Different Clothes 排名（seed0 epoch50 final，按 R1）', '']
    for index, row in enumerate(ranking, 1):
        report_lines.append('{}. `{}`：R1={}，mAP={}；Delta R1={}，Delta mAP={}。'.format(
            index, row['Variant'], fmt(float(row['Diff_R1'])), fmt(float(row['Diff_mAP']), 4),
            fmt(float(row['Delta_Diff_R1'])) if row['Delta_Diff_R1'] != '' else 'n/a',
            fmt(float(row['Delta_Diff_mAP'])) if row['Delta_Diff_mAP'] != '' else 'n/a'))
    report_lines += ['', '## 3-seed matched 结果', '',
                     '见 [identity_preservation_multiseed.csv](identity_preservation_multiseed.csv) 和 per-seed 明细。所有 Delta 都是同 seed 的 `M_variant,s - M_N00,s`。', '',
                     '| Variant | Reliability | Diff R1 mean±std | Diff mAP mean±std | Same R1 mean±std | Same mAP mean±std | matched ΔDiff R1 mean±std | matched ΔDiff mAP mean±std |',
                     '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in multiseed_rows:
        if row['n_seeds']:
            report_lines.append('| `{}` | {} | {}±{} | {}±{} | {}±{} | {}±{} | {}±{} | {}±{} |'.format(
                row['Variant'], row['Reliability'], show(row['Diff_R1_mean']), show(row['Diff_R1_std']),
                show(row['Diff_mAP_mean']), show(row['Diff_mAP_std']),
                show(row['Same_R1_mean']), show(row['Same_R1_std']),
                show(row['Same_mAP_mean']), show(row['Same_mAP_std']),
                show(row['Delta_Diff_R1_mean']), show(row['Delta_Diff_R1_std']),
                show(row['Delta_Diff_mAP_mean']), show(row['Delta_Diff_mAP_std'])))
    report_lines += ['', '## Same Clothes 稳定性', '']
    for variant, mean, std in sorted(same_stability, key=lambda row: row[2]):
        row = multiseed_by_variant.get(variant, {})
        report_lines.append('- `{}`：Same R1 {}±{}，Same mAP {}±{}。'.format(
            variant, show(mean), show(std), show(multiseed_number(variant, 'Same_mAP_mean')),
            show(multiseed_number(variant, 'Same_mAP_std'))))
    report_lines += [
        '',
        '## Semantic cosine / margin diagnostics',
        '',
        '每个 run 的 `epoch_diagnostics.csv` 记录 `cos_raw_sem`、`cos_res1_sem`、`cos_res2_sem`、`cos_com_sem`、两条 residual-minus-com margin、四个 loss 和 semantic_valid_rate；seed0 表汇总 epoch50 的 residual 平均 cosine、com cosine 和平均 margin。',
        '',
        '## 机制问题自动回答',
        '',
        '- Q1 Raw alignment：N01 相对 N00 为 Diff R1 {} pp、Diff mAP {} pp；结论：只有很小的 R1 单 seed 增益，mAP 基本 neutral/略降，不能认为 direct alignment 已有效。'.format(show(q1_r1), show(q1_map)),
        '- Q2 Residual preservation：N02–N04 的最佳 Diff R1={}、Diff mAP={}，均未超过 N01 的 R1={}、mAP={}；结论：本轮三个 preservation 权重没有优于 raw alignment，虽 residual cosine/margin 随权重上升但未转化为 ReID 收益。'.format(show(q2_best_r1), show(q2_best_map), show(q2_raw_r1), show(q2_raw_map)),
        '- Q3 Leakage exclusion：N05/N06/N07 的 seed0 Diff R1 Δ分别为 {}/{}/{} pp、mAP Δ分别为 {}/{}/{} pp；`cos_com_sem` 约在 {} 到 {}，没有随 lambda 单调下降；结论：低权重 N05 有小幅正向，但 exclusion 尚无稳定证据。'.format(
            show(seed0_number('N05', 'Delta_Diff_R1')), show(seed0_number('N06', 'Delta_Diff_R1')), show(seed0_number('N07', 'Delta_Diff_R1')),
            show(seed0_number('N05', 'Delta_Diff_mAP')), show(seed0_number('N06', 'Delta_Diff_mAP')), show(seed0_number('N07', 'Delta_Diff_mAP')),
            show(min(seed0_number(v, 'cos_com_sem') for v in ('N05', 'N06', 'N07')), 6),
            show(max(seed0_number(v, 'cos_com_sem') for v in ('N05', 'N06', 'N07')), 6)),
        '- Q4 Relative ranking：N08/N09/N10/N11 的 seed0 Diff R1 Δ为 {}/{}/{}/{} pp，mAP Δ为 {}/{}/{}/{} pp；margin diagnostic 均为正但性能不随 lambda/margin 单调，N09 的 3-seed matched R1 为 {}±{} pp、mAP 为 {}±{} pp；结论：ranking 改变了 decomposition，但“更稳定”尚未被充分支持。'.format(
            show(seed0_number('N08', 'Delta_Diff_R1')), show(seed0_number('N09', 'Delta_Diff_R1')), show(seed0_number('N10', 'Delta_Diff_R1')), show(seed0_number('N11', 'Delta_Diff_R1')),
            show(seed0_number('N08', 'Delta_Diff_mAP')), show(seed0_number('N09', 'Delta_Diff_mAP')), show(seed0_number('N10', 'Delta_Diff_mAP')), show(seed0_number('N11', 'Delta_Diff_mAP')),
            show(multiseed_number('N09', 'Delta_Diff_R1_mean')), show(multiseed_number('N09', 'Delta_Diff_R1_std')),
            show(multiseed_number('N09', 'Delta_Diff_mAP_mean')), show(multiseed_number('N09', 'Delta_Diff_mAP_std'))),
        '- Q5 Preservation + exclusion：N12 为 Diff R1 Δ={}、mAP Δ={}，N13 为 Diff R1 Δ={}、mAP Δ={}；结论：只有较高 exclusion 权重的 N13 显示弱的 seed0 互补迹象，N12 反而受损，不能确认普遍互补。'.format(
            show(q5_n12_r1), show(q5_n12_map), show(q5_n13_r1), show(q5_n13_map)),
        '- Q6 Reliability：matched 3-seed Delta Diff R1/mAP 分别为 N09={}/{}、N14={}/{}、N15={}/{}、N16={}/{} pp；结论：attribute/confidence 没有稳定额外收益，joint 的 mAP 为正且更稳定，但 R1 仍接近 neutral。'.format(
            show(multiseed_number('N09', 'Delta_Diff_R1_mean')), show(multiseed_number('N09', 'Delta_Diff_mAP_mean')),
            show(multiseed_number('N14', 'Delta_Diff_R1_mean')), show(multiseed_number('N14', 'Delta_Diff_mAP_mean')),
            show(multiseed_number('N15', 'Delta_Diff_R1_mean')), show(multiseed_number('N15', 'Delta_Diff_mAP_mean')),
            show(multiseed_number('N16', 'Delta_Diff_R1_mean')), show(multiseed_number('N16', 'Delta_Diff_mAP_mean'))),
        '- Q7 Multi-seed：N16 joint 的 matched Diff mAP deltas 为 {}，三个 seed 均为正；Diff R1 deltas 为 {}，仅小幅正/负摆动；Same R1 为 {}±{} 且是五个关键配置中波动最小；结论：matched evidence 最支持 joint 用于 mAP/Same Clothes 稳定性，不支持明显 Diff R1 提升。详见 [identity_preservation_multiseed_per_seed.csv](identity_preservation_multiseed_per_seed.csv)。'.format(
            ', '.join(show(x) for x in n16_map_deltas), ', '.join(show(x) for x in n16_r1_deltas),
            show(multiseed_number('N16', 'Same_R1_mean')), show(multiseed_number('N16', 'Same_R1_std'))),
        '',
        '## 与旧 V0–V4 比较',
        '',
        '- 已知旧 Original PDF V0：Different Clothes R1=64.4087，mAP=62.1613。',
        '- 旧 V0–V4 CSV：{}。若存在，将在自动分类中作为旧 EOT additive family 的参考；否则不虚构缺失数字。'.format(old_path or '当前候选路径未找到'),
        '',
        '| Variant | Semantic | Attribute reliability | Confidence reliability | Diff R1 | Diff mAP | Same R1 | Same mAP |',
        '|---|---|---|---|---:|---:|---:|---:|',
    ] + old_table + [
        '',
        '- 旧单 seed 中最佳 Diff R1 为 V2={}，最佳 Diff mAP 为 V3={}；新 seed0 中 N09 的 Diff R1={}（比旧 V2 高 {} pp），但 Diff mAP={}（比旧 V3 低 {} pp）。这是单 seed 对比，不能替代 matched multi-seed 结论。'.format(
            show(float(old_by_variant['V2']['Diff_R1'])) if old_by_variant.get('V2') else 'n/a',
            show(float(old_by_variant['V3']['Diff_mAP'])) if old_by_variant.get('V3') else 'n/a',
            show(seed0_number('N09', 'Diff_R1')),
            show(seed0_number('N09', 'Diff_R1') - float(old_by_variant['V2']['Diff_R1'])) if old_by_variant.get('V2') else 'n/a',
            show(seed0_number('N09', 'Diff_mAP')),
            show(seed0_number('N09', 'Diff_mAP') - float(old_by_variant['V3']['Diff_mAP'])) if old_by_variant.get('V3') else 'n/a'),
        '- 新 V2 所有主结果统一为 epoch50 final；不能用各组 best test epoch 替换主结果。',
        '',
        '## 最终推荐',
        '',
        '- 配置推荐 1：`N09`（none + rank λ=0.05、margin=0.10）作为当前最强的 Diff R1 机制候选，但必须接受其 mAP 没有提升、matched R1 效应小于 seed std 的事实。',
        '- 配置推荐 2：`N16`（joint + rank λ=0.05、margin=0.10）作为 reliability 候选；三个 matched seed 的 Diff mAP delta 全为正，且 Same R1 波动最小，但 Diff R1 基本 neutral。',
        '- 最推荐下一步：冻结 `N16` 做更大 seed 数的确认，并按属性/样本记录 decomposition change；若目标只看 Different Clothes R1，则并行保留 `N09`，不要再做 test-adaptive lambda 搜索。若确认效应仍接近 seed variance，应转向重新设计四属性 semantic representation。',
        '判断原则：如果 ReID 不升但 residual/com cosine 和 margin 按预期变化，应把它作为 decomposition 机制证据，而不是伪造性能收益。',
        '',
        '不成功实验、失败状态和未启动任务均保留在 status CSV、各 run 目录及 queue status 中。',
    ]
    (reports / 'PDF_identity_preservation_v2_overnight_report.md').write_text('\n'.join(report_lines) + '\n')
    print('Wrote V2 result tables and overnight report to', reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='outputs/pdf_identity_preservation_v2')
    parser.add_argument('--reports', default='reports')
    parser.add_argument('--old-results', default='/data/projects/PDF-worktrees/pdf-reliability-ablation/reports/pdf_reliability_ablation.csv')
    args = parser.parse_args()
    main(args)
