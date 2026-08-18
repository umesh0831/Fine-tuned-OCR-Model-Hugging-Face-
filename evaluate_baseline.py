#!/usr/bin/env python3
"""
Baseline (and later, fine-tuned) evaluation for the prescription OCR test split.

Computes, per test image:
  - CER / WER (secondary, general transcription quality)
  - Dosage-pattern match (proxy for critical-field accuracy, since ground
    truth is free-text rather than structured fields)
  - Document-type classification (Has Medication: yes/no) accuracy

Fill in `predict()` with your model's inference call, then run this
inside your Lightning Studio / Colab / Kaggle environment where the
model is loaded.

Usage:
    python3 evaluate_baseline.py --csv data/test_ground_truth.csv --images data/test
"""
import argparse
import csv
import os
import re

try:
    import jiwer
except ImportError:
    raise SystemExit("Run: pip install jiwer")

DOSAGE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s?(?:mg|mcg|ml|mL|g|gm|IU|units?)\b", re.IGNORECASE
)


def predict(image_path: str) -> str:
    """
    Plug in your model's inference call here.
    Must return the model's raw transcription of the image as a string.
    """
    raise NotImplementedError("Wire this up to your DeepSeek-OCR inference call")


def extract_dosages(text: str) -> set:
    return {m.group(0).replace(" ", "").lower() for m in DOSAGE_PATTERN.finditer(text)}


def evaluate(csv_path: str, images_dir: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    cer_scores, wer_scores = [], []
    dosage_correct, dosage_total = 0, 0
    doctype_correct, doctype_total = 0, 0
    per_image = []

    for row in rows:
        fname = row["Filename"]
        gt_text = row["Extracted Text"]
        has_med_gt = row.get("Has Medication", "Yes") == "Yes"

        image_path = os.path.join(images_dir, fname)
        pred_text = predict(image_path)

        cer = jiwer.cer(gt_text, pred_text)
        wer = jiwer.wer(gt_text, pred_text)
        cer_scores.append(cer)
        wer_scores.append(wer)

        gt_dosages = extract_dosages(gt_text)
        pred_dosages = extract_dosages(pred_text)
        if has_med_gt:
            dosage_total += 1
            # correct if every ground-truth dosage string appears in the prediction
            if gt_dosages and gt_dosages.issubset(pred_dosages):
                dosage_correct += 1

        has_med_pred = len(pred_dosages) > 0
        doctype_total += 1
        if has_med_pred == has_med_gt:
            doctype_correct += 1

        per_image.append({
            "filename": fname,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "gt_dosages": sorted(gt_dosages),
            "pred_dosages": sorted(pred_dosages),
        })

    n = len(rows)
    print(f"Test set size: n = {n}")
    print(f"Mean CER: {sum(cer_scores)/n:.4f}")
    print(f"Mean WER: {sum(wer_scores)/n:.4f}")
    if dosage_total:
        print(f"Dosage-pattern match accuracy: {dosage_correct}/{dosage_total} "
              f"({100*dosage_correct/dosage_total:.1f}%)")
    print(f"Document-type (has medication) accuracy: {doctype_correct}/{doctype_total} "
          f"({100*doctype_correct/doctype_total:.1f}%)")

    return per_image


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    args = ap.parse_args()
    evaluate(args.csv, args.images)
