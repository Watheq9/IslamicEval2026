# IslamicEval 2026 — Submission Instructions

## 1. Prepare your prediction file

Each subtask expects **one TSV file** (tab-separated, with a header row):

| Task | Filename (any name is fine) | Required columns |
|------|------------------------------|-------------------|
| Task 1 – Segmentation | `*.tsv` | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Span_Start`, `Span_End` |
| Task 2 – Labelling | `*.tsv` | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Label` (`correct` / `incorrect`) |
| Task 3 – Correction | `*.tsv` | `Response_ID`, `Annotation_ID`, `Segment_Type`, `Correction` |
| Task 4 – Relevance | `*.tsv` | `question_id`, `Response_ID`, `Annotation_ID`, `span_type`, `span_text`, `relevance_label` |

Make sure:
- The file is UTF-8 encoded.
- Columns are tab-separated and match the names above exactly.
- Each `(Response_ID, Annotation_ID, Segment_Type)` (or `question_id` triple for Task 4) appears **only once**.

## 2. Zip your file

Codabench requires a **`.zip` archive containing your `.tsv` file** — not the raw `.tsv` itself, and not a folder.

```bash
zip submission.zip your_predictions.tsv
```

Your `.zip` should contain the `.tsv` file directly at the top level (no subfolders):

```
submission.zip
└── your_predictions.tsv
```

## 3. Submit on Codabench

1. Go to the competition page and open the **My Submissions** tab for the relevant phase (e.g. Test or Practice).
2. Click **Submit**, and upload your `submission.zip`.
3. Wait for the run to finish — status will change from *Submitted* → *Running* → **Finished** (success) or **Failed** (an issue with your file's format).
4. If a submission **Fails**, check the run log for a specific error message (e.g. a missing column or invalid label) and re-submit a corrected file.

## Notes
- You may submit multiple times during the test/devlopement phase; only your selected submission counts toward the leaderboard.
- Scores may be hidden during certain phases — a "Finished" status confirms your file was valid and scored, even if you can't see the number yet.
- Questions? Contact the organizers via the competition forum or email listed on the competition page.