# IslamicEval 2026

This repository contains the **dataset**, **corpora**, and **evaluation scripts** for [**IslamicEval 2026** — a shared task on Fine-grained Hallucination Detection
in Arabic Islamic Content](https://sites.google.com/view/islamiceval2026/). The shared task consists of four subtasks for detecting, verifying, correcting, and assessing the relevance of Qur'anic and Hadith citations in LLM-generated Arabic responses to Islamic questions.

Responses are drawn from real user questions (sourced from Fanar logs) answered by large language models. Each response may cite Qur'anic verses (Ayah) or Prophetic sayings (Hadith), and every citation has been manually annotated at a fine-grained level: is the quoted text authentic, is the chain of narration (isnad) correct, is the claimed source accurate, and — if something is wrong — what is the correct citation?

The task is split into four subtasks:

| Subtask | Name | Goal |
|---|---|---|
| 1 | Span Detection | Locate citation spans (Ayah, matn, isnad, claimed_source) within a response |
| 2 | Hallucination Identification | Classify each segment as `correct` or `incorrect` |
| 3 | Correction | Provide the authentic text for incorrect Ayah/matn citations |
| 4 | Answer Relevance | Determine whether a correct/corrected citation is relevant to the question it answers |

## Repository structure

```
IslamicEval2026/
├── Corpora/               # Reference corpora used to ground/verify citations
│                           #   (quranic_verses.json, six_hadith_books.json)
├── Scoring_scripts/        # Official evaluation scripts (task1_scoring.py … task4_scoring.py)
├── train_set/              # Training data (responses + annotations)
├── dev_set/                # Development / validation data
├── submission_examples/                # Examples on dev set to follow when submitting to Codabench
├── CODEBOOK.md              # Field-by-field schema of the JSONL annotation format
├── DATASET_README.md        # Full dataset description, submission format, and scoring details
├── submission_README.md     # How to package and submit predictions to Codabench
└── README.md                 # You are here
```

## Dataset at a glance

- ~5,000 model-generated Arabic responses to real Islamic questions
- 15,000+ annotated citations, ~33,000 labelled segments
- Each citation is decomposed into typed segments: `Ayah`, `matn`, `isnad`, `claimed_source`
- Each segment is labelled `correct`, `incorrect`, or `N/A` (isnad/claimed_source are only judged when the underlying Ayah/matn is correct)
- Incorrect Ayah/matn segments carry one or more accepted **corrections** (or the literal string `خطأ` when the citation cannot be grounded in any authentic source)
- Responses that cite nothing are explicitly marked (`no_annotation`)
- All Qur'anic verses and Hadiths are grounded against the provided reference corpora, `Corpora/quranic_verses.json` and `Corpora/six_hadith_books.json`

See **[DATASET_README.md](./DATASET_README.md)** for the complete dataset description and submission file formats, and **[CODEBOOK.md](./CODEBOOK.md)** for the exact JSON schema of every field in `train.jsonl` / `dev.jsonl`.

## Submission format (summary)

Predictions are submitted as tab-separated (TSV) files, one per subtask:

| Subtask | Columns |
|---|---|
| 1 — Span Detection | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Span_Start`, `Span_End` |
| 2 — Hallucination Identification | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Label` |
| 3 — Correction | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Correction` |
| 4 — Answer Relevance | `question_id`, `Response_ID`, `Annotation_ID`, `span_type`, `span_text`, `relevance_label` |

Full column definitions, worked examples, and edge cases (e.g. no-citation responses, multiple accepted corrections) are documented in [DATASET_README.md](./DATASET_README.md). Packaging instructions for the official submission are in [submission_README.md](./submission_README.md).

We provide examples on the dev set per task in "submission_examples" dir. The goal is to make it easier for you to understand what we expect from you as an input on Codabench. You can use these toy examples in the development phase to test the submission on Codabench platform.

## Running the evaluation scripts

Each subtask has its own scorer in `Scoring_scripts/`: `task1_scoring.py`, `task2_scoring.py`, `task3_scoring.py`, `task4_scoring.py`. All are run the same way.

### Method 1 — folder convention

1. Open the relevant script and set `ROOT_DIR` at the top (or set the `SCORING_ROOT` environment variable).
2. Inside that folder, create:
   ```
   input/ref/    # gold answer file(s) (+ the responses .jsonl, for Subtask 1 only)
   input/res/    # your single prediction .tsv
   output/       # the script writes scores.json here
   ```
3. Run:
   ```bash
   python task1_scoring.py
   python task2_scoring.py
   python task3_scoring.py
   python task4_scoring.py
   ```

### Method 2 — explicit paths

```bash
python task1_scoring.py --pred my_preds.tsv --ref gold.tsv --responses responses.jsonl --output ./out
python task2_scoring.py --pred my_preds.tsv --ref gold.tsv --output ./out
python task3_scoring.py --pred my_preds.tsv --ref gold.tsv --output ./out
python task4_scoring.py --pred my_preds.tsv --ref gold.tsv --output ./out
```

### Metrics reported

- **Subtask 1:** character-level macro-F1 over five classes (`neither`, `Ayah`, `matn`, `isnad`, `claimed_source`)
- **Subtask 2:** accuracy per segment type, macro-averaged (gold `N/A` excluded)
- **Subtask 3:** overall accuracy, plus per-type accuracy for `Ayah` and `matn`
- **Subtask 4:** macro-averaged F1 across questions, with per-question precision/recall/F1 breakdown

## Requirements

The evaluation scripts are pure Python (no external dependencies beyond the standard library, unless noted in the individual scripts). A recent Python 3 installation is sufficient to run them.

## Task website

Detailed task descriptions, including the Subtask 4 relevance-label guidelines, are published at:
https://sites.google.com/view/islamiceval2026/

## Citation

If you use this dataset or evaluation scripts in your work, please cite the IslamicEval 2026 shared task (citation details to be added soon insha'Allah).

## License

The dataset and corpora are available for research purposes only.
