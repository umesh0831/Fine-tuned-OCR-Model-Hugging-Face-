---
base_model: unsloth/DeepSeek-OCR
library_name: peft
tags:
  - ocr
  - lora
  - vision-language
  - medical
  - prescription
  - unsloth
---

# DeepSeek-OCR — Medical Prescription Fine-Tune (LoRA)

LoRA fine-tune of [unsloth/DeepSeek-OCR](https://huggingface.co/unsloth/DeepSeek-OCR) (3B params)
for transcribing handwritten medical prescriptions, with a focus on correctly identifying
**drug name and dosage** — the safety-critical fields in this document type — not just overall
text similarity.

## Why field-level accuracy, not just CER/WER

A blended text-similarity score (like CER) treats a typo in a patient's address the same as a
typo in "500mg" vs "50mg." For a medical document, those are not equally serious. This project
reports CER/WER **and** structured drug-name / dosage accuracy separately, scored only against
manually-confirmed ground truth.

## Training details

- **Base model:** unsloth/DeepSeek-OCR (3B), loaded via Unsloth, `load_in_4bit=False`
- **Method:** LoRA (`r=16`, `lora_alpha=16`, target modules: q/k/v/o/gate/up/down_proj) —
  77,509,632 trainable params (2.27% of total)
- **Training data:** 131 real handwritten prescription images, free-text transcription target
- **Steps:** 60 (max_steps), effective batch size 8 (`per_device_train_batch_size=1`,
  `gradient_accumulation_steps=8`), `paged_adamw_8bit` optimizer, `lr=2e-4`
- **Training loss:** ~1.9 (step 1) → ~0.55 (step 60)

## Evaluation results (test set, n=32 images)

| Metric | Base model (no LoRA) | Fine-tuned (this model) | Change |
|---|---|---|---|
| Mean CER ↓ | 0.8469 | 0.9629 | worse (+0.116) |
| Mean WER ↓ | 1.2027 | 1.1666 | slightly better |
| Doc-type (has medication) accuracy ↑ | 25.0% (8/32) | 43.8% (14/32) | +18.8pp |
| Critical-field drug-name accuracy ↑ | 2.1% (1/47) | 14.9% (7/47) | ~7x better |
| Critical-field drug+dosage accuracy ↑ | 0.0% (0/47) | 8.5% (4/47) | 0% → 8.5% |

*Critical-field accuracy is scored only against test-set rows manually confirmed clean
(84% of the test split's drug/dosage rows).*

### Reading these numbers honestly

Fine-tuning on this small dataset (131 images, 60 steps) **improved the metrics that matter most
for this task** — drug-name and dosage identification — while raw whole-transcription CER stayed
roughly flat (slightly worse). The likely explanation: with limited training data, the LoRA
adapter learned to bias output toward the structure/vocabulary of medication lines specifically
(what the training labels emphasize), at a small cost to fidelity on surrounding free text
(patient info, doctor notes) that CER measures indiscriminately across the whole output.

## Limitations

- **Not validated for clinical or production use.** Drug+dosage accuracy is 8.5% — this model
  gets the safety-critical field right a small minority of the time. It is a research/portfolio
  fine-tune demonstrating an evaluation methodology, not a deployable medical tool.
- Trained on only 131 images — small dataset, high variance expected on out-of-distribution
  handwriting styles.
- Base model still shows repetition/runaway-generation outliers on some inputs (see project
  `STATUS.md` for details); not fully addressed in this fine-tune.

## Usage

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "Rahul3736/deepseek-ocr-medical-prescription",
    load_in_4bit=False,
    auto_model="AutoModel",
    trust_remote_code=True,
)
FastVisionModel.for_inference(model)

output = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR. ",
    image_file="your_prescription.jpg",
    base_size=1024,
    image_size=640,
    crop_mode=True,
)
```
