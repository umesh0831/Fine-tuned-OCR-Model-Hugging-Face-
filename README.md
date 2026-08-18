# OCR Fine-Tuning Eval — Prescription OCR with DeepSeek-OCR (LoRA)

Evaluates a LoRA fine-tune of [DeepSeek-OCR (3B)](https://huggingface.co/unsloth/DeepSeek-OCR)
for transcribing handwritten medical prescriptions — with a focus on correctly identifying
**drug name and dosage**, the safety-critical fields, not just overall text similarity.

Fine-tuned model: [huggingface.co/Rahul3736/deepseek-ocr-medical-prescription](https://huggingface.co/Rahul3736/deepseek-ocr-medical-prescription)

## Why field-level accuracy, not just CER/WER

A blended text-similarity score (like Character Error Rate) treats a typo in a patient's address
the same as a typo in "500mg" vs "50mg." For a medical document those are not equally serious, so
this project reports CER/WER **and** structured drug-name/dosage accuracy separately — scored only
against manually-confirmed ground truth.

## Results (test set, n=32 images)

| Metric | Base model (no LoRA) | Fine-tuned | Change |
|---|---|---|---|
| Mean CER ↓ | 0.8469 | 0.9629 | worse (+0.116) |
| Mean WER ↓ | 1.2027 | 1.1666 | slightly better |
| Doc-type (has medication) accuracy ↑ | 25.0% (8/32) | 43.8% (14/32) | +18.8pp |
| Critical-field drug-name accuracy ↑ | 2.1% (1/47) | 14.9% (7/47) | ~7x better |
| Critical-field drug+dosage accuracy ↑ | 0.0% (0/47) | 8.5% (4/47) | 0% → 8.5% |

Fine-tuning on a small dataset (131 images, 60 steps) improved the metrics that matter most for
this task — drug-name and dosage identification — while raw whole-transcription CER stayed roughly
flat. Full write-up, including all training challenges (OOM fixes, kernel-state debugging) and an
explain-it-simply breakdown, is in [`STATUS.md`](./STATUS.md).

**Not validated for clinical or production use** — see Limitations in
[`HF_MODEL_CARD.md`](./HF_MODEL_CARD.md).

## Repo layout

```
ocr-finetune-eval/
├── data/                          # prescription images + ground truth (CC0-1.0 licensed)
│   ├── train/ val/ test/          # images
│   ├── *_ground_truth.csv         # free-text transcription per image
│   └── *_critical_fields.csv      # structured Drug Name/Dosage, long format (one row per drug)
├── kaggle_baseline_eval.ipynb     # main notebook: baseline eval -> LoRA fine-tune -> after eval
├── Deepseek_OCR_(3B)_Eval.ipynb   # original Unsloth template notebook
├── label_tool.py                  # manual transcription labeling helper
├── STATUS.md                      # full project log: decisions, challenges, results
└── HF_MODEL_CARD.md               # model card mirrored to the Hugging Face repo
```

## Method

- **Base model:** `unsloth/DeepSeek-OCR` (3B), loaded via Unsloth
- **Fine-tuning:** LoRA (`r=16`, `lora_alpha=16`), 77.5M trainable params (2.27% of total)
- **Training data:** 131 handwritten prescription images, 60 steps, effective batch size 8
- **Evaluation:** CER/WER (via `jiwer`) + structured critical-field accuracy, scored only against
  manually-reviewed ground truth (84% of the test split's drug/dosage rows are confirmed clean)

See [`STATUS.md`](./STATUS.md) for the full session log, including how the ground-truth labeling
was done, bugs found and fixed in the field-extraction parser, and every training issue hit along
the way (CUDA OOM, missing imports, kernel-state debugging) with plain-language explanations.
