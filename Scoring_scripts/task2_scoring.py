

import argparse
import json
import os
from collections import defaultdict
import pandas as pd

'''
IslamicEval 2026 - Task 2 (Labelling) scorer.
Per-segment-type accuracy, EXCLUDING gold-N/A rows.

Input files:
1- Place a .TSV file of the gold solutions in {ref} directory (or pass it by command line using --ref argument)
2- Place your prediction TSV file in {res} directory (or pass it by command line using --pred argument)
3- Place {ref} and {res} directories in a directory namde {input}

Prediction TSV columns (tab-separated, WITH header):
    Response_ID   Annotation_ID   Segment_Type   Label
Label must be only "correct" or "incorrect".
Note (deviation from 2025): predictions carry id columns and are matched by
(Response_ID, Annotation_ID, Segment_Type), NOT by row order, because each
citation has multiple segments and positional matching would misalign.
Rows where the GOLD label is "N/A" are excluded from scoring.
 
Usage:
    python task2_scoring.py
        (uses the default competition directories below)
    OR
    python task2_scoring.py --pred my_preds.tsv --ref gold.tsv --output ./out
        (uses explicit files/dirs instead)
'''


ROOT_DIR = '/app/'
DEFAULT_REFERENCE_DIR = os.path.join(ROOT_DIR, 'input/', 'ref')
DEFAULT_PREDICTION_DIR = os.path.join(ROOT_DIR, 'input/', 'res')
DEFAULT_SCORE_DIR = os.path.join(ROOT_DIR, 'output/')
 
REQUIRED_COLUMNS = ('Response_ID', 'Annotation_ID', 'Segment_Type', 'Label')
KEY_COLUMNS = ['Response_ID', 'Annotation_ID', 'Segment_Type']
VALID_PRED_LABELS = {'correct', 'incorrect'}
 

def parse_args():
    parser = argparse.ArgumentParser(
        description='IslamicEval 2026 - Task 2 (Labelling) scorer.'
    )
    parser.add_argument(
        '--pred', '-p', default=None,
        help=f'Prediction TSV file, or a directory containing exactly one '
             f'.tsv file (default: {DEFAULT_PREDICTION_DIR})'
    )
    parser.add_argument(
        '--ref', '-r', default=None,
        help=f'Reference TSV file, or a directory containing exactly one '
             f'.tsv file (default: {DEFAULT_REFERENCE_DIR})'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help=f'Directory to write scores.json into (default: {DEFAULT_SCORE_DIR})'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help='Print the computed scores to stdout. Use this flag if you want to print output on stdout'
    )
    return parser.parse_args()
 


 
def resolve_tsv_file(path, label):
    """Accept either a direct file path or a directory containing a single
    (or several, in which case the alphabetically-first is used) .tsv file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} path does not exist: {path}')
 
    if os.path.isfile(path):
        return path
 
    # It's a directory: find tsv files inside it.
    candidates = sorted(f for f in os.listdir(path) if f.endswith('.tsv'))
    if not candidates:
        raise FileNotFoundError(
            f'No {label.lower()} file found in "{path}". '
            f'Ensure there is exactly one file in TSV format.'
        )
    if len(candidates) > 1:
        print(
            f'WARNING: multiple .tsv files found in "{path}": {candidates}. '
            f'Using "{candidates[0]}". Pass an explicit file path to avoid ambiguity.'
        )
    chosen = candidates[0]
    print(f'Found {label.lower()} file: {chosen}')
    return os.path.join(path, chosen)
 

 
 
def load_tsv(path):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    # Normalize whitespace so stray leading/trailing spaces don't break key matching.
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df
 

def validate_columns(df, name):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f'{name} file must contain column(s): {", ".join(missing_cols)}.'
        )
 
 
def validate_prediction_labels(pred_data):
    bad_rows = pred_data.index[~pred_data['Label'].isin(VALID_PRED_LABELS)].tolist()
    if bad_rows:
        bad_values = pred_data.loc[bad_rows, 'Label'].unique().tolist()
        raise ValueError(
            f'Prediction file "Label" column must contain only "correct" or '
            f'"incorrect". Found invalid value(s) {bad_values} at row(s) {bad_rows}.'
        )
 
 
def build_pred_lookup(pred_data):
    lookup = {}
    duplicates = []
    for _, r in pred_data.iterrows():
        k = tuple(r[c] for c in KEY_COLUMNS)
        if k in lookup:
            duplicates.append(k)
        lookup[k] = r['Label']
    if duplicates:
        raise ValueError(
            f'Prediction file contains duplicate rows for the same '
            f'(Response_ID, Annotation_ID, Segment_Type) key(s): {duplicates[:10]}'
            f'{" ..." if len(duplicates) > 10 else ""}. Each combination must '
            f'appear exactly once.'
        )
    return lookup
 
 
def score(pred_data, ref_data, verbose=False):
    validate_columns(pred_data, 'Prediction')
    validate_columns(ref_data, 'Reference')
    validate_prediction_labels(pred_data)
 
    pred_lookup = build_pred_lookup(pred_data)
 
    # Exclude gold N/A rows from scoring.
    ref_scored = ref_data[ref_data['Label'] != 'N/A']
 
    per_type = defaultdict(lambda: [0, 0])
    missing_per_type = defaultdict(int)
    missing_total = 0
 
    for _, r in ref_scored.iterrows():
        k = tuple(r[c] for c in KEY_COLUMNS)
        seg_type = r['Segment_Type']
        per_type[seg_type][1] += 1
        if k in pred_lookup:
            if pred_lookup[k] == r['Label']:
                per_type[seg_type][0] += 1
        else:
            missing_total += 1
            missing_per_type[seg_type] += 1
 
    per_type_acc = {t: c / n for t, (c, n) in sorted(per_type.items()) if n > 0}
    macro = sum(per_type_acc.values()) / len(per_type_acc) if per_type_acc else 0.0
 
    scores = {'accuracy': macro}
    scores.update({f'accuracy_{t}': a for t, a in per_type_acc.items()})
    scores['missing_predictions'] = missing_total
    scores.update({
        f'missing_predictions_{t}': n for t, n in sorted(missing_per_type.items())
    })
 
    if missing_total and verbose:
        print(
            f'WARNING: {missing_total} gold row(s) had no matching prediction '
            f'(scored as incorrect): {dict(sorted(missing_per_type.items()))}'
        )
 
    return scores
 
 
def main():
    args = parse_args()
 
    pred_path = args.pred if args.pred is not None else DEFAULT_PREDICTION_DIR
    ref_path = args.ref if args.ref is not None else DEFAULT_REFERENCE_DIR
    score_dir = args.output if args.output is not None else DEFAULT_SCORE_DIR
 
    print('Reading prediction')
    pred_file = resolve_tsv_file(pred_path, 'Prediction')
    pred_data = load_tsv(pred_file)
 
    print('Reading reference')
    ref_file = resolve_tsv_file(ref_path, 'Reference')
    ref_data = load_tsv(ref_file)
 
    print('Checking Accuracy')
    scores = score(pred_data, ref_data, verbose=args.verbose)
 
    os.makedirs(score_dir, exist_ok=True)
    out_path = os.path.join(score_dir, 'scores.json')
    with open(out_path, 'w') as score_file:
        score_file.write(json.dumps(scores, indent=2))
    
    if args.verbose:
        print('Scores:')
        print(scores)
    else:
        print('Submission processed and scored successfully.')


if __name__ == '__main__':
    main()
 







