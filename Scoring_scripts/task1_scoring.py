import argparse
import json
import os

import pandas as pd
from sklearn.metrics import f1_score

'''
IslamicEval 2026 - Task 1 (Segmentation) scorer.
Corpus-pooled character-level macro-F1 over 5 classes:
    neither, Ayah, matn, isnad, claimed_source

Input files:
1- Place the reference JSONL (responses) and the gold TSV in {ref} directory
   (or pass them by command line using --ref / --responses)
2- Place your prediction TSV in {res} directory (or pass it via --pred)
3- Place {ref} and {res} directories in a directory named {input}

Prediction TSV columns (tab-separated, WITH header):
    Response_ID   Annotation_ID   Segment_Type   Span_Start   Span_End
Note (deviation from 2025): predictions carry id columns and are matched by
(Response_ID, Annotation_ID, Segment_Type), NOT by row order, because each
citation has multiple segments and positional matching would misalign.
Spans are character offsets into generated_answer, end-exclusive.

Usage:
    python task1_scoring.py
        (uses the default competition directories below)
    OR
    python task1_scoring.py --pred my_preds.tsv --ref gold.tsv --responses responses.jsonl --output ./out
        (uses explicit files/dirs instead)
'''

ROOT_DIR = '/app/'
DEFAULT_REFERENCE_DIR = os.path.join(ROOT_DIR, 'input/', 'ref')
DEFAULT_PREDICTION_DIR = os.path.join(ROOT_DIR, 'input/', 'res')
DEFAULT_SCORE_DIR = os.path.join(ROOT_DIR, 'output/')

CLASSES = {'neither': 0, 'Ayah': 1, 'matn': 2, 'isnad': 3, 'claimed_source': 4}
ALLOWED_SEGMENT_TYPES = set(CLASSES) | {'NoAnnotation'}
REQUIRED_SPAN_COLUMNS = ('Response_ID', 'Annotation_ID', 'Segment_Type', 'Span_Start', 'Span_End')


def parse_args():
    parser = argparse.ArgumentParser(
        description='IslamicEval 2026 - Task 1 (Segmentation) scorer.'
    )
    parser.add_argument(
        '--pred', '-p', default=None,
        help=f'Prediction TSV file, or a directory containing exactly one '
             f'.tsv file (default: {DEFAULT_PREDICTION_DIR})'
    )
    parser.add_argument(
        '--ref', '-r', default=None,
        help=f'Gold spans TSV file, or a directory containing exactly one '
             f'.tsv file (default: {DEFAULT_REFERENCE_DIR})'
    )
    parser.add_argument(
        '--responses', default=None,
        help=f'Reference JSONL (responses) file, or a directory containing '
             f'exactly one .jsonl file (default: {DEFAULT_REFERENCE_DIR})'
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


def resolve_file(path, extension, label):
    """Accept either a direct file path or a directory containing a single
    (or several, in which case the alphabetically-first is used) file with
    the given extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} path does not exist: {path}')

    if os.path.isfile(path):
        return path

    candidates = sorted(f for f in os.listdir(path) if f.lower().endswith(extension))
    if not candidates:
        raise FileNotFoundError(
            f'No {label.lower()} file found in "{path}". '
            f'Ensure there is exactly one {extension} file.'
        )
    if len(candidates) > 1:
        print(
            f'WARNING: multiple {extension} files found in "{path}": {candidates}. '
            f'Using "{candidates[0]}". Pass an explicit file path to avoid ambiguity.'
        )
    chosen = candidates[0]
    print(f'Found {label.lower()} file: {chosen}')
    return os.path.join(path, chosen)


def load_tsv(path):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


def load_responses(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{line_no}: invalid JSON ({e})') from e
            if 'id' not in rec or 'generated_answer' not in rec:
                raise ValueError(
                    f'{path}:{line_no}: record must contain "id" and "generated_answer" keys.'
                )
            mapping[rec['id']] = rec['generated_answer']
    if not mapping:
        raise ValueError(f'{path}: no records found.')
    return mapping


def validate_columns(df, name):
    missing = [c for c in REQUIRED_SPAN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{name} file must contain column(s): {", ".join(missing)}.')


def validate_segment_types(df, name):
    bad_rows = df.index[~df['Segment_Type'].isin(ALLOWED_SEGMENT_TYPES)].tolist()
    if bad_rows:
        bad_values = df.loc[bad_rows, 'Segment_Type'].unique().tolist()
        raise ValueError(
            f'{name} file "Segment_Type" column must contain only '
            f'{", ".join(sorted(ALLOWED_SEGMENT_TYPES))}. '
            f'Found invalid value(s) {bad_values} at row(s) {bad_rows}.'
        )


def paint(rows, length, name):
    arr = [0] * length
    for idx, r in rows.iterrows():
        typ = r['Segment_Type']
        raw_start = str(r['Span_Start']).strip()
        if typ == 'NoAnnotation' or raw_start in ('-', '', 'nan'):
            continue
        try:
            start, end = int(r['Span_Start']), int(r['Span_End'])
        except ValueError as e:
            raise ValueError(
                f'{name} row {idx}: Span_Start/Span_End must be integers '
                f'(got {r["Span_Start"]!r}, {r["Span_End"]!r}).'
            ) from e
        start = max(0, start)
        cls = CLASSES.get(typ, 0)
        for i in range(start, min(end, length)):
            arr[i] = cls
    return arr


def score(qid_response_mapping, ref_data, pred_data, verbose=False):
    validate_columns(ref_data, 'Reference')
    validate_columns(pred_data, 'Prediction')
    validate_segment_types(ref_data, 'Reference')
    validate_segment_types(pred_data, 'Prediction')

    unmatched_pred_ids = set(pred_data['Response_ID']) - set(qid_response_mapping.keys())

    g_all, p_all = [], []
    for rid, text in qid_response_mapping.items():
        length = len(text)
        g_all.extend(paint(ref_data[ref_data['Response_ID'] == rid], length, 'Reference'))
        p_all.extend(paint(pred_data[pred_data['Response_ID'] == rid], length, 'Prediction'))

    labels = list(CLASSES.values())
    f1 = round(f1_score(g_all, p_all, average='macro', labels=labels, zero_division=0), 4)

    scores = {'F1': f1}

    if unmatched_pred_ids and verbose:
        print(
            f'WARNING: prediction file contains {len(unmatched_pred_ids)} Response_ID(s) '
            f'not present among the reference responses; those rows were ignored: '
            f'{sorted(unmatched_pred_ids)[:10]}'
        )

    return scores


def main():
    args = parse_args()

    pred_path = args.pred if args.pred is not None else DEFAULT_PREDICTION_DIR
    ref_path = args.ref if args.ref is not None else DEFAULT_REFERENCE_DIR
    responses_path = args.responses if args.responses is not None else DEFAULT_REFERENCE_DIR
    score_dir = args.output if args.output is not None else DEFAULT_SCORE_DIR

    print('Reading reference responses (JSONL)')
    resp_file = resolve_file(responses_path, '.jsonl', 'Responses')
    qid_response_mapping = load_responses(resp_file)
    print(f'Reference responses loaded with {len(qid_response_mapping)} entries')

    print('Reading reference spans (TSV)')
    ref_file = resolve_file(ref_path, '.tsv', 'Reference')
    ref_data = load_tsv(ref_file)
    print(f'Reference spans was loaded successfully!')

    print('Reading prediction')
    pred_file = resolve_file(pred_path, '.tsv', 'Prediction')
    pred_data = load_tsv(pred_file)

    print('Calculating F1 Score')
    scores = score(qid_response_mapping, ref_data, pred_data, verbose=args.verbose)

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
