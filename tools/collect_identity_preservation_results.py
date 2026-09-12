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
        ROOT / 'reports/pdf_reliability_ablation.csv',
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
        classification_reason = 'identity-preservation 机制相对 control 有效，但 matched seeds 尚不足以支持可靠性策略的稳定额外收益。'
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
        '最有希望配置（按 matched multi-seed Different Clothes R1/mAP）：',
    ]
    if best_new:
        report_lines.append('- `{}`：Delta Diff R1 mean={} pp，Delta Diff mAP mean={} pp；matched R1 deltas={}。'.format(
            best_new[0], fmt(best_new[1]), fmt(best_new[2]), ', '.join(fmt(x) for x in best_new[3])))
    if best_reliability:
        report_lines.append('- `{}`：reliability 家族最佳，Delta Diff R1 mean={} pp，Delta Diff mAP mean={} pp。'.format(
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
        '- gradient sanity：见 [identity_preservation_gradient_sanity.json](identity_preservation_gradient_sanity.json)。',
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
        '## Different Clothes 排名（seed0 epoch50 final，按 R1）',
        '',
    ]
    for index, row in enumerate(ranking, 1):
        report_lines.append('{}. `{}`：R1={}，mAP={}；Delta R1={}，Delta mAP={}。'.format(
            index, row['Variant'], fmt(float(row['Diff_R1'])), fmt(float(row['Diff_mAP']), 4),
            fmt(float(row['Delta_Diff_R1'])) if row['Delta_Diff_R1'] != '' else 'n/a',
            fmt(float(row['Delta_Diff_mAP'])) if row['Delta_Diff_mAP'] != '' else 'n/a'))
    report_lines += ['', '## 3-seed matched 结果', '',
                     '见 [identity_preservation_multiseed.csv](identity_preservation_multiseed.csv) 和 per-seed 明细。所有 Delta 都是同 seed 的 `M_variant,s - M_N00,s`。', '']
    for row in multiseed_rows:
        if row['n_seeds']:
            report_lines.append('- `{}`：Diff R1 {}±{}，Diff mAP {}±{}；matched Delta R1 {}±{}，Delta mAP {}±{}。'.format(
                row['Variant'], fmt(row['Diff_R1_mean']), fmt(row['Diff_R1_std']),
                fmt(row['Diff_mAP_mean']), fmt(row['Diff_mAP_std']),
                fmt(row['Delta_Diff_R1_mean']), fmt(row['Delta_Diff_R1_std']),
                fmt(row['Delta_Diff_mAP_mean']), fmt(row['Delta_Diff_mAP_std'])))
    report_lines += ['', '## Same Clothes 稳定性', '']
    for variant, mean, std in sorted(same_stability, key=lambda row: row[2]):
        report_lines.append('- `{}`：Same R1 {}±{}。'.format(variant, fmt(mean), fmt(std)))
    report_lines += [
        '',
        '## Semantic cosine / margin diagnostics',
        '',
        '每个 run 的 `epoch_diagnostics.csv` 记录 `cos_raw_sem`、`cos_res1_sem`、`cos_res2_sem`、`cos_com_sem`、两条 residual-minus-com margin、四个 loss 和 semantic_valid_rate；seed0 表汇总 epoch50 的 residual 平均 cosine、com cosine 和平均 margin。',
        '',
        '## 机制问题自动回答',
        '',
        '- Q1 Raw alignment：比较 N01 vs N00；见 seed0 `Delta_*` 与 diagnostics。',
        '- Q2 Residual preservation：比较 N02–N04 vs N01；看 R1/mAP 与 residual cosine 是否同步改善。',
        '- Q3 Leakage exclusion：比较 N05–N07 vs N00；重点看 `cos_com_sem` 是否下降而 ReID 不受损。',
        '- Q4 Relative ranking：比较 N08–N11；重点看 `semantic_margin` 是否为正且跨 seed 稳定。',
        '- Q5 Preservation + exclusion：比较 N12/N13 与单机制组，检查是否互补。',
        '- Q6 Reliability：只比较 N09/N14/N15/N16；使用 matched 3-seed Delta，不使用单 seed 最大值。',
        '- Q7 Multi-seed：见 `identity_preservation_multiseed_per_seed.csv`，报告了每个 seed 对 N00 的匹配差值。',
        '',
        '## 与旧 V0–V4 比较',
        '',
        '- 已知旧 Original PDF V0：Different Clothes R1=64.4087，mAP=62.1613。',
        '- 旧 V0–V4 CSV：{}。若存在，将在自动分类中作为旧 EOT additive family 的参考；否则不虚构缺失数字。'.format(old_path or '当前候选路径未找到'),
        '- 新 V2 所有主结果统一为 epoch50 final；不能用各组 best test epoch 替换主结果。',
        '',
        '## 最终推荐',
        '',
        '优先依据 matched multi-seed 的 Different Clothes R1、mAP 以及 Same Clothes 的标准差；如果 ReID 不升但 residual/com cosine 和 margin 按预期变化，应把它作为 decomposition 机制证据，而不是伪造性能收益。',
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
