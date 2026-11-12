"""Reproduce the eight updated result tables from saved answers (no API calls).

Run from the repository root: python -m src.evaluation.export_paper_results
The CSVs retain full numeric precision; the plain LaTeX exports are for numeric
reproduction. The reviewed manuscript layout/coloring lives in updated_paper_tables.md.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from .eval_tool import load_all_questions, load_answers, enrich_answers, analysis, summary_xidk
from .eval_relation import load_relations, load_relation_clf, merge_relations_by_action
from .summary_idk_worst import summary_idk_worst, impute_worst_case
from .eval_pvalue import compute_pvals
from .export_supplementary_tables import build_tables, render_table

ROOT = Path(__file__).resolve().parents[2]
BINARY = ['?A1=A2', '?A1>A3', '?A1>A4', '?A3∅A4', '?A4=A1|3']
SIM = ['J(A1-A2)', 'J(A3-A4)', 'J(A4-A1|3)']
DISPLAY = ['?A1=A1*', 'J(A1-A1*)'] + BINARY + ['CON'] + SIM + ['idk']
TABLES = {
    'main': 'tab:consistency_per_action_by_id',
    'filtered': 'tab:consistency_per_action_filtering',
    'classification': 'tab:classification',
    'significance': 'tab:p-value_prompting',
    'correlation': 'tab:Pearson',
    'cfr': 'tab:cRi_by_id_act', 'correct': 'tab:cRi+', 'incorrect': 'tab:cRi-',
}
STRATEGIES = ['zero-shot', 'classification', 'fixing']


def export_reviewed_tables(markdown, outdir, selected, base_dir, init_base=False,
                           current_dir=None, main_tex=None, appendix_tex=None):
    """Compare reviewed tables with a frozen original-paper baseline."""
    from .table_revisions import freeze_base, export_versions
    if current_dir is not None:
        blocks = [(current_dir / (key + '.tex')).read_text() for key in TABLES]
    elif markdown.exists():
        text = markdown.read_text()
        blocks = re.findall(r'^```latex[ \t]*\n(.*?)^```[ \t]*$', text, re.M | re.S)
    else:
        source_dir = ROOT / 'output/paper_tables'
        blocks = [(source_dir / (key + '.tex')).read_text() for key in TABLES]
    by_label = {}
    for block in blocks:
        labels = re.findall(r'\\label\{([^}]+)\}', block)
        if len(labels) != 1 or labels[0] not in TABLES.values():
            raise ValueError('Each LaTeX block must have exactly one recognized table label')
        label = labels[0]
        if label in by_label:
            raise ValueError(f'Duplicate table label: {label}')
        if not re.fullmatch(r'\\begin\{(table\*?)\}[\s\S]*\\end\{\1\}\s*', block):
            raise ValueError(f'Incomplete table environment: {label}')
        by_label[label] = block
    missing = set(TABLES.values()) - by_label.keys()
    if missing:
        raise ValueError(f'Missing table labels: {sorted(missing)}')
    if init_base:
        freeze_base(base_dir, main_tex, appendix_tex, TABLES)
    elif not (base_dir / 'manifest.json').exists():
        raise ValueError('No frozen baseline found. Initialize it once using --source reviewed --init-base')
    export_versions({key: by_label[label] for key, label in TABLES.items()},
                    base_dir, outdir, selected, TABLES)


def rebuild(root, models):
    datasets = ['LC-QuAD', 'qawiki', 'spinach', 'synthetic']
    tasks = ['equal', 'sup-sub', 'minus']
    questions = load_all_questions(str(root), datasets, ['en'])
    answers = load_answers(str(root / 'data/answers'), datasets, models,
        STRATEGIES + ['star'], tasks, ['en'], ['Q1', 'Q2', 'Q3', 'Q4'])
    frame = analysis(enrich_answers(answers, questions))
    relations = load_relations(str(root / 'data/answers/zero-shot'), datasets, models)
    labels = load_relation_clf(str(root / 'data/answers/CtE'), datasets, models, tasks)
    return merge_relations_by_action(frame, relations, labels), relations


def percent_summary(frame, filtered=False):
    out = (summary_xidk if filtered else summary_idk_worst)(frame)
    out = out[out.dataset.eq('overall') & out.action.isin(STRATEGIES)].copy()
    out['CON'] = out[BINARY].mean(axis=1)
    # The repeated-question control is displayed in Base rows only.
    out.loc[~out.action.eq('zero-shot'), DISPLAY[:2]] = float('nan')
    for col in DISPLAY:
        out[col] *= 100
    return out


def simple_tex(data, label):
    """Plain reproducible table; publication layout is kept separately."""
    def escape(value):
        return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
    def cell(value):
        if pd.isna(value):
            return '--'
        if isinstance(value, (float, int)):
            return f'{value:.3g}' if label == TABLES['correlation'] else f'{value:.2f}'
        return escape(value)
    lines = [r'\begin{table*}[!t]', r'\centering\scriptsize',
             r'\begin{tabular}{' + 'l' * len(data.columns) + '}', r'\toprule',
             ' & '.join(map(escape, data.columns)) + r' \\', r'\midrule']
    lines += [' & '.join(map(cell, row)) + r' \\' for row in data.itertuples(index=False, name=None)]
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{Recomputed results.}',
              r'\label{' + label + '}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def export(root, outdir, selected):
    inputs = sorted(p for p in (root / 'data').rglob('*')
                    if p.is_file() and p.suffix in {'.json', '.csv'})
    def fingerprints():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in inputs}
    manifest = fingerprints()
    info = json.loads((root / 'data/llm_info.json').read_text())
    models = sorted(info, key=lambda model: info[model]['ID'])
    frame, relations = rebuild(root, models)
    outdir.mkdir(parents=True, exist_ok=True)
    main = percent_summary(frame)
    results = {'main': main, 'filtered': percent_summary(frame, filtered=True)}
    truths = ['Equivalence', 'Contains', 'Contains', 'Disjoint', 'Equivalence']
    relcols = ['R(1-2)', 'R(1-3)', 'R(1-4)', 'R(3-4)', 'R(1-34)']
    rows = []
    for model, group in relations.groupby('llm'):
        row = {'llm': model, 'n_items': len(group)}
        for col, truth in zip(relcols, truths):
            row[col] = 100 * group[col].eq(truth).mean()
            row[col + '_n'] = int(group[col].notna().sum())
        row['average'] = sum(row[col] for col in relcols) / 5
        rows.append(row)
    results['classification'] = pd.DataFrame(rows)
    pvals = compute_pvals(impute_worst_case(frame))
    results['significance'] = pvals[pvals.dataset.eq('overall')
        & pvals.action.isin(['classification', 'fixing'])].copy()
    base = main[main.action.eq('zero-shot')].set_index('llm')
    eligible = [m for m in models if info[m].get('score') is not None]
    rows = []
    for metric in BINARY + ['CON'] + SIM:
        r, p = pearsonr([info[m]['score'] for m in eligible], base.loc[eligible, metric])
        rows.append({'metric': metric, 'r': r, 'p': p, 'n_models': len(eligible)})
    results['correlation'] = pd.DataFrame(rows)
    supplementary = build_tables(frame, models)
    for key, number in [('cfr', 9), ('correct', 10), ('incorrect', 11)]:
        results[key] = supplementary[supplementary.table.eq(number)].copy()
    current_inputs = sorted(p for p in (root / 'data').rglob('*')
                            if p.is_file() and p.suffix in {'.json', '.csv'})
    if inputs != current_inputs or manifest != fingerprints():
        raise RuntimeError('Input data changed during evaluation; rerun after data updates finish.')
    (outdir / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Per-item sources and denominators support auditing the two repeated-Q1 columns.
    base_rows = frame[frame.action.eq('zero-shot')]
    base_rows[['llm', 'dataset', 'Q_ID', 'repeat_source', '?A1=A1*', 'J(A1-A1*)']].to_csv(outdir / 'repeat_provenance.csv', index=False)
    frame.groupby(['llm', 'action']).size().rename('n_instances').to_csv(outdir / 'coverage.csv')
    for key in selected:
        data = results[key]
        data.to_csv(outdir / (key + '.csv'), index=False)
        if key in ['cfr', 'correct', 'incorrect']:
            number = {'cfr': 9, 'correct': 10, 'incorrect': 11}[key]
            tex = render_table(supplementary, number, models, 'raw')
        else:
            if key in ['main', 'filtered']:
                view = data[['llm', 'action'] + DISPLAY]
            elif key == 'classification':
                view = data[['llm'] + relcols + ['average']]
            elif key == 'significance':
                view = data[['llm', 'action'] + [m.replace('?', 'p(') + ')' for m in BINARY]].copy()
                for col in view.columns[2:]:
                    view[col] = view[col].map(lambda p: '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '--')
            else:
                view = data
            tex = simple_tex(view, TABLES[key])
        (outdir / (key + '.tex')).write_text(tex)
        print(key, '->', outdir / (key + '.csv'), '/', key + '.tex')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', choices=['all'] + list(TABLES), default='all')
    parser.add_argument('--source', choices=['raw', 'reviewed'], default='raw',
                        help='raw: recompute scores; reviewed: generate clean/blue tables against the original-paper baseline')
    parser.add_argument('--markdown', type=Path, default=ROOT / 'output/updated_paper_tables.md',
                        help='Reviewed Markdown input for --source reviewed')
    parser.add_argument('--current-dir', type=Path,
                        help='Read updated .tex tables from this directory instead of Markdown')
    parser.add_argument('--base-dir', type=Path, default=ROOT / 'output/paper_tables/base',
                        help='Frozen original-paper baseline directory')
    parser.add_argument('--init-base', action='store_true',
                        help='Initialize the baseline once from the original manuscript; refuses to overwrite it')
    parser.add_argument('--base-main', type=Path, default=ROOT / 'TEMP/ICLR.tex')
    parser.add_argument('--base-appendix', type=Path, default=ROOT / 'TEMP/appendix.tex')
    parser.add_argument('--output-dir', type=Path,
                        help='Default: output/reproduced_tables (raw) or output/paper_tables (reviewed)')
    args = parser.parse_args()
    if args.source != 'reviewed' and (args.init_base or args.current_dir):
        parser.error('--init-base and --current-dir require --source reviewed')
    selected = list(TABLES) if args.table == 'all' else [args.table]
    outdir = args.output_dir or ROOT / 'output' / (
        'paper_tables' if args.source == 'reviewed' else 'reproduced_tables')
    if args.source == 'reviewed':
        export_reviewed_tables(args.markdown, outdir, selected, args.base_dir,
                               args.init_base, args.current_dir, args.base_main, args.base_appendix)
    else:
        export(ROOT, outdir, selected)


if __name__ == '__main__':
    main()
