#!/usr/bin/env python3
"""Evaluate submitted predications to Task 4 - IslamicEval 2026 """

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path



'''
IslamicEval 2026 - Task 4 (Labelling) scorer.

1- Place a TSV of the expected solutions in {reference_dir}
2- Place your prediction TSV in {prediction_dir}

Prediction TSV columns (tab-separated, WITH header):
    question_id   Response_ID   Annotation_ID   span_type   span_text   relevance_label

Given a user question and an LLM response with its correctly extracted Qur'anic and Hadith citation spans, 
the task is to determine whether each citation span is relevant to answering the question. 
The question_id, Response_ID, and Annotation_ID fields link to the main JSONL file, which contains the 
full text of the user's question and the LLM's response.
span_type identifies the citation as either an Ayah or a matn.
span_text provides the correct text of the citation.
The task is to predict the relevance_label for each span using a binary classification:
 Relevant (1) or Non-relevant (0). 

Refer to the detailed relevance label definitions published at Subtask 4 webpage https://sites.google.com/view/islamiceval2026/subtask-4

'''


EXPECTED_COLUMNS = ["question_id", "Response_ID", "Annotation_ID", "span_type", "span_text", "relevance_label"]
KEY_COLUMNS = ("question_id", "Response_ID", "Annotation_ID")
OUTPUT_COLUMNS = ["question_id", "Response_ID", "span_count", "tp", "fp", "fn", "tn", "precision", "recall", "F1 Score"]
DEFAULT_OUTPUT = Path("task4_question_level_scores.tsv")

def binary_label(value):
    value = value.strip()
    if value in {"0", "0.0"}: return 0
    if value in {"1", "1.0", "2", "2.0"}: return 1
    raise ValueError(f"unsupported relevance_label {value!r}; expected 0, 1, or 2")

def validate_tsv(path):
    if path.suffix.lower() != ".tsv": raise ValueError(f"{path}: input must have a .tsv extension")
    try: path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error: raise ValueError(f"{path}: input is not valid UTF-8") from error
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_COLUMNS: raise ValueError(f"{path}: columns must be exactly, in order: " + ", ".join(EXPECTED_COLUMNS))
        records, seen = [], set()
        for row_number, row in enumerate(reader, 2):
            if None in row or any(row[column] is None for column in EXPECTED_COLUMNS): raise ValueError(f"{path}:{row_number}: incorrect number of TSV columns")
            key = tuple(row[column].strip() for column in KEY_COLUMNS)
            if any(not value for value in key): raise ValueError(f"{path}:{row_number}: blank identifier value")
            if key in seen: raise ValueError(f"{path}:{row_number}: duplicate key {'/'.join(key)}")
            seen.add(key); record = {column: row[column].strip() for column in EXPECTED_COLUMNS}
            try: record["binary_label"] = binary_label(row["relevance_label"])
            except ValueError as error: raise ValueError(f"{path}:{row_number}: {error}") from error
            records.append(record)
    if not records: raise ValueError(f"{path}: no data rows")
    return records

def row_key(record): return tuple(str(record[column]) for column in KEY_COLUMNS)

def align_rows(gold, predictions):
    gold_by_key = {row_key(row): row for row in gold}; prediction_by_key = {row_key(row): row for row in predictions}
    missing = sorted(set(gold_by_key) - set(prediction_by_key))
    extra = sorted(set(prediction_by_key) - set(gold_by_key))
    if missing or extra:
        details = []
        if missing: details.append(f"missing {len(missing)} prediction row(s), e.g. {'/'.join(missing[0])}")
        if extra: details.append(f"extra {len(extra)} prediction row(s), e.g. {'/'.join(extra[0])}")
        raise ValueError("; ".join(details))
    return [(gold_by_key[key], prediction_by_key[key]) for key in sorted(gold_by_key)]

def question_scores(pairs):
    grouped = defaultdict(list)
    for gold, prediction in pairs: grouped[str(gold["question_id"])].append((gold, prediction))
    results, no_relevant_count = [], 0
    for question_id in sorted(grouped):
        question_pairs, counts = grouped[question_id], Counter()
        for gold, prediction in question_pairs:
            actual, predicted = int(gold["binary_label"]), int(prediction["binary_label"])
            if actual == 1 and predicted == 1: counts["tp"] += 1
            elif actual == 0 and predicted == 1: counts["fp"] += 1
            elif actual == 1 and predicted == 0: counts["fn"] += 1
            else: counts["tn"] += 1
        tp, fp, fn, tn = (counts[name] for name in ("tp", "fp", "fn", "tn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if tp + fn == 0: no_relevant_count += 1; f1 = 1.0 if fp == 0 else 0.0
        else: f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append({"question_id": question_id, "response_ids": ", ".join(sorted({str(g["response_id"]) for g, _ in question_pairs})), "span_count": len(question_pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "F1 Score": f1})
    return results, no_relevant_count

def evaluate(gold_path, prediction_path, output_path):
    rows, count = question_scores(align_rows(validate_tsv(gold_path), validate_tsv(prediction_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    return sum(float(row["F1 Score"]) for row in rows) / len(rows)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gold", type=Path, required=True)
    parser.add_argument("-p", "--predictions", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try: 
        macro_f1 = evaluate(args.gold, args.predictions, args.output)
    except (OSError, ValueError) as error: 
        raise SystemExit(f"ERROR: {error}") from error
    
    # result = {}
    print(f"Macro-averaged F1: {macro_f1:.6f}")
    print(f"Saved question-level F1 scores: {args.output}")
    return macro_f1

if __name__ == "__main__": main()