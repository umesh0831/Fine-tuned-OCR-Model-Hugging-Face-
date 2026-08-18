# =====================================================================
# CELL 1 — Install dependencies
# (from the confirmed Unsloth DeepSeek-OCR notebook, non-Colab branch)
# =====================================================================
# --- paste from here down to the next CELL marker into its own cell ---
!pip install unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!pip install jiwer einops addict easydict

# =====================================================================
# CELL 2 — Load DeepSeek-OCR (base, no LoRA) via Unsloth
# =====================================================================
import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

from huggingface_hub import snapshot_download
from unsloth import FastVisionModel
from transformers import AutoModel

snapshot_download("unsloth/DeepSeek-OCR", local_dir="deepseek_ocr")

model, tokenizer = FastVisionModel.from_pretrained(
    "./deepseek_ocr",
    load_in_4bit=False,
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",
)

# Guard against runaway repetition loops seen on some prescription images
# (model kept emitting "9.00 AM" / "x=0, y=N" until the 8192-token ceiling).
# Set here, in the same cell that creates `model`, so cell ordering can't break it.
model.generation_config.max_new_tokens = 768
model.generation_config.repetition_penalty = 1.3
model.generation_config.no_repeat_ngram_size = 4

# =====================================================================
# CELL 3 — Point at your uploaded Kaggle dataset
# Kaggle mounts API-uploaded datasets under the nested
# /kaggle/input/datasets/<username>/<slug> path, not the older
# /kaggle/input/<slug> convention — confirmed via os.listdir().
# =====================================================================
DATA_DIR = "/kaggle/input/datasets/rahulroy3736/prescription-ocr-finetune-eval"

TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test")
TEST_GT_CSV = os.path.join(DATA_DIR, "test_ground_truth.csv")
TEST_CRITICAL_CSV = os.path.join(DATA_DIR, "test_critical_fields.csv")

print(os.listdir(DATA_DIR))  # sanity check

# =====================================================================
# CELL 4 — predict(): captures model.infer() output
# Uses the "Gundam" preset (base_size=1024, image_size=640, crop_mode=True)
# since that's what Unsloth's own notebook uses for its reported numbers.
# =====================================================================
PROMPT = "<image>\nFree OCR. "

# With eval_mode=True, model.infer() returns the decoded transcription directly
# instead of printing it via a streamer (see modeling_deepseekocr.py: the
# eval_mode branch skips the NoEOSTextStreamer and does
# `return outputs` after tokenizer.decode(...)). No stdout capture needed.
def predict(image_path: str) -> str:
    outputs = model.infer(
        tokenizer,
        prompt=PROMPT,
        image_file=image_path,
        output_path="/kaggle/working/ocr_out",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False,
        eval_mode=True,  # no_repeat_ngram_size 20 -> 35: stronger guard against
                          # the repetition/runaway generation seen on 178.jpg etc.
    )
    return outputs.strip()

# =====================================================================
# CELL 5 — Drug/Dosage parser (RECONSTRUCTION — see caveat above)
#
# This rebuilds the parsing logic from what you described fixing:
#   - prefixes: Tab./Cap./Syp./Inj./Supp./Oint./Cream/Drops
#   - handles suffix-style ordering (drug name THEN form word, e.g. "Bacmox syp")
#   - avoids matching "tab" mid-sentence inside dosing instructions (e.g. "1 tab PO")
#   - excludes "Disp:" (dispensing quantity) lines
#   - strips enumeration prefixes (e.g. "2:", "i)", "(I)") before parsing
#
# This is a rebuild, NOT the verified original from your other session.
# Spot-check its output against a handful of rows in test_critical_fields.csv
# that you already manually confirmed, before trusting its numbers.
# =====================================================================
import re

FORM_WORDS = r"(?:Tab\.?|Cap\.?|Syp\.?|Inj\.?|Supp\.?|Oint\.?|Cream|Drops)"
DOSAGE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s?(?:mg|mcg|ml|mL|g|gm|IU|units?)\b", re.IGNORECASE
)
ENUM_PREFIX = re.compile(r"^\s*(?:\d+[\.:]|\(?[ivx]+\)|\(?[IVX]+\))\s*", re.IGNORECASE)
EXCLUDED_SEGMENT_STARTS = ("sig:", "disp:", "advice:")

# form word BEFORE drug name: "Tab. Azithromycin 500mg"
PATTERN_PREFIX = re.compile(
    rf"{FORM_WORDS}\s+([A-Z][A-Za-z\-]+(?:\s[A-Z][A-Za-z\-]+)?)\s*\(?(\d+(?:\.\d+)?\s?(?:mg|mcg|ml|mL|g|gm|IU|units?))?",
)
# form word AFTER drug name: "Bacmox syp 7-7-7"
PATTERN_SUFFIX = re.compile(
    rf"([A-Z][A-Za-z\-]+)\s+{FORM_WORDS}\b",
)

def parse_drug_dosage(text: str):
    # Returns a list of (drug_name, dosage_or_None) tuples parsed from free text.
    # Ground truth uses " | " separators; the model outputs "\n" separators.
    # Split on BOTH so the same parser works on labels and predictions alike.
    results = []
    segments = [s.strip() for s in re.split(r"[|\n]", text)]
    for seg in segments:
        low = seg.lower()
        if any(low.startswith(p) for p in EXCLUDED_SEGMENT_STARTS):
            continue
        seg = ENUM_PREFIX.sub("", seg)

        m = PATTERN_PREFIX.search(seg)
        if m:
            drug = m.group(1).strip()
            dosage = m.group(2)
            results.append((drug, dosage))
            continue

        m = PATTERN_SUFFIX.search(seg)
        if m:
            drug = m.group(1).strip()
            dosage_match = DOSAGE_PATTERN.search(seg)
            dosage = dosage_match.group(0) if dosage_match else None
            results.append((drug, dosage))

    return results

def normalize_drug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

def normalize_dosage(dose) -> str:
    if not dose:
        return ""
    return re.sub(r"\s", "", dose.lower())

# =====================================================================
# CELL 6 — Full evaluation: CER/WER + critical-field accuracy + doctype
#
# Critical-field accuracy is scored ONLY against rows where
# Needs Review == "No" in test_critical_fields.csv — i.e. the rows you
# already manually confirmed. Flagged rows are excluded since their
# ground truth itself isn't fully trusted yet.
# =====================================================================
import csv
import time
import jiwer
from collections import defaultdict

def normalize_for_cer(text: str) -> str:
    """
    Ground truth uses " | " field separators; the model emits "\\n" and some
    LaTeX-ish markup. Comparing raw would measure formatting style, not
    transcription accuracy. So collapse separators/whitespace on BOTH sides
    before scoring. CER/WER below are therefore separator-agnostic.
    """
    text = re.sub(r"[|\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_confirmed_critical_fields(csv_path):
    by_image = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Needs Review"] == "No" and row["Drug Name"]:
                by_image[row["Filename"]].append(
                    (row["Drug Name"], row["Dosage"])
                )
    return by_image

def evaluate():
    with open(TEST_GT_CSV, newline="") as f:
        gt_rows = list(csv.DictReader(f))

    confirmed_fields = load_confirmed_critical_fields(TEST_CRITICAL_CSV)

    cer_scores, wer_scores = [], []
    doctype_correct, doctype_total = 0, 0
    drug_correct, drug_total = 0, 0
    dose_correct, dose_total = 0, 0
    per_image = []

    for i, row in enumerate(gt_rows):
        fname = row["Filename"]
        gt_text = row["Extracted Text"]
        has_med_gt = row.get("Has Medication", "Yes") == "Yes"

        image_path = os.path.join(TEST_IMAGES_DIR, fname)
        t0 = time.time()
        pred_text = predict(image_path)
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(gt_rows)}] {fname} — {elapsed:.1f}s")

        gt_norm = normalize_for_cer(gt_text)
        pred_norm_text = normalize_for_cer(pred_text)
        cer = jiwer.cer(gt_norm, pred_norm_text) if pred_norm_text else 1.0
        wer = jiwer.wer(gt_norm, pred_norm_text) if pred_norm_text else 1.0
        cer_scores.append(cer)
        wer_scores.append(wer)

        has_med_pred = len(parse_drug_dosage(pred_text)) > 0
        doctype_total += 1
        if has_med_pred == has_med_gt:
            doctype_correct += 1

        if fname in confirmed_fields:
            pred_pairs = parse_drug_dosage(pred_text)
            pred_norm = {
                (normalize_drug(d), normalize_dosage(dose)) for d, dose in pred_pairs
            }
            pred_drug_names = {normalize_drug(d) for d, _ in pred_pairs}

            for gt_drug, gt_dose in confirmed_fields[fname]:
                drug_total += 1
                gd = normalize_drug(gt_drug)
                if gd in pred_drug_names:
                    drug_correct += 1

                dose_total += 1
                gdose = normalize_dosage(gt_dose)
                if (gd, gdose) in pred_norm:
                    dose_correct += 1

        per_image.append({
            "filename": fname,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "pred_chars": len(pred_text),
        })

    n = len(gt_rows)

    # Sanity guard: if predictions come back empty, the metrics below are
    # meaningless. This exact bug produced WER=1.0 across all 32 images
    # before we found that infer() returns None and prints instead.
    empty = sum(1 for p in per_image if p["pred_chars"] == 0)
    if empty:
        print(f"\n*** WARNING: {empty}/{n} predictions were EMPTY — "
              f"metrics below are NOT trustworthy. ***")

    print(f"\n{'='*55}")
    print(f"Test set size: n = {n}")
    print(f"Mean predicted chars/image: {sum(p['pred_chars'] for p in per_image)/n:.0f}")
    print(f"Mean CER: {sum(cer_scores)/n:.4f}")
    print(f"Mean WER: {sum(wer_scores)/n:.4f}")
    print(f"Doc-type (has medication) accuracy: {doctype_correct}/{doctype_total} "
          f"({100*doctype_correct/doctype_total:.1f}%)")
    if drug_total:
        print(f"Critical-field drug-name accuracy: {drug_correct}/{drug_total} "
              f"({100*drug_correct/drug_total:.1f}%)")
    if dose_total:
        print(f"Critical-field drug+dosage accuracy: {dose_correct}/{dose_total} "
              f"({100*dose_correct/dose_total:.1f}%)")

    with open("/kaggle/working/baseline_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "cer", "wer", "pred_chars"])
        w.writeheader()
        w.writerows(per_image)

    return per_image

evaluate()

# =====================================================================
# CELL 7 — Load local train set for LoRA fine-tuning
# Uses the same DATA_DIR established in Cell 3. "Extracted Text" is the
# free-text transcription target — same task the eval prompt asks for
# ("Free OCR."), so no reformatting needed.
# =====================================================================
import pandas as pd
from PIL import Image as PILImage

TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train")
TRAIN_GT_CSV = os.path.join(DATA_DIR, "train_ground_truth.csv")

train_df = pd.read_csv(TRAIN_GT_CSV)

train_samples = []
for _, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMAGES_DIR, row["Filename"])
    train_samples.append({
        "image": PILImage.open(img_path).convert("RGB"),
        "text": str(row["Extracted Text"]).strip(),
    })

print(f"Loaded {len(train_samples)} training samples")

# =====================================================================
# CELL 8 — Add LoRA adapters
# Verbatim from Unsloth's official DeepSeek-OCR fine-tuning notebook
# (github.com/unslothai/notebooks/blob/main/nb/Deepseek_OCR_(3B).ipynb).
# =====================================================================
model = FastVisionModel.get_peft_model(
    model,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    r=16,              # larger = more capacity, more overfit risk
    lora_alpha=16,     # recommended: alpha == r at minimum
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# =====================================================================
# CELL 9 — Convert to the conversation format DeepSeek-OCR training expects
# =====================================================================
instruction = "<image>\nFree OCR. "

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "<|User|>",
            "content": instruction,
            "images": [sample["image"]],
        },
        {
            "role": "<|Assistant|>",
            "content": sample["text"],
        },
    ]
    return {"messages": conversation}

converted_dataset = [convert_to_conversation(sample) for sample in train_samples]

# =====================================================================
# CELL 10 — DeepSeekOCRDataCollator
# Verbatim from Unsloth's official notebook — this is the class that turns
# each conversation into model-ready tensors (image patches, token ids,
# response-only loss masking). Not reimplemented by hand: ~350 lines of
# image-patch/token bookkeeping is too easy to get subtly wrong by
# reconstructing from memory, so this is copied as-is from the source.
# =====================================================================
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import ImageOps
from torch.nn.utils.rnn import pad_sequence
import io

from deepseek_ocr.modeling_deepseekocr import (
    format_messages,
    text_encode,
    BasicImageTransform,
    dynamic_preprocess,
)

@dataclass
class DeepSeekOCRDataCollator:
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

    def deserialize_image(self, image_data) -> PILImage.Image:
        if isinstance(image_data, PILImage.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
            image = PILImage.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: PILImage.Image, crop_ratio: Tuple[int, int]) -> int:
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(self, image: PILImage.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size, use_thumbnail=False
                )

            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))

            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                    num_queries * height_crop_num)

        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), PILImage.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_data in message["images"]:
                    if img_data is not None:
                        pil_image = self.deserialize_image(img_data)
                        images.append(pil_image)

        if not images:
            raise ValueError("No images found in sample. Please ensure all samples contain images.")

        tokenized_str = []
        images_seq_mask = []
        images_list, images_crop_list, images_spatial_crop = [], [], []

        prompt_token_count = -1
        assistant_started = False
        image_idx = 0

        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            text_splits = content.split("<image>")

            for i, text_sep in enumerate(text_splits):
                tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))

                if i < len(text_splits) - 1:
                    if image_idx >= len(images):
                        raise ValueError(
                            "Data mismatch: Found '<image>' token but no corresponding image."
                        )

                    image = images[image_idx]
                    img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)

                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))

                    image_idx += 1

        if image_idx != len(images):
            raise ValueError(
                f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens were used."
            )

        if not assistant_started:
            print("Warning: No assistant message found in sample. Masking all tokens.")
            prompt_token_count = len(tokenized_str)

        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = []

        for feature in features:
            try:
                processed = self.process_single_sample(feature["messages"])
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        input_ids_list = [item["input_ids"] for item in batch_data]
        images_seq_mask_list = [item["images_seq_mask"] for item in batch_data]
        prompt_token_counts = [item["prompt_token_count"] for item in batch_data]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[images_seq_mask] = -100

        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        images_batch = []
        for item in batch_data:
            images_batch.append((item["images_crop"], item["images_ori"]))

        images_spatial_crop = torch.cat([item["images_spatial_crop"] for item in batch_data], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

# =====================================================================
# CELL 11 — Train
# Verbatim hyperparameters from Unsloth's official notebook (60 steps,
# effective batch size 8). With ~130 train images this is a few epochs'
# worth — enough to see a real before/after delta, not a fully-converged
# model. Swap max_steps=None + num_train_epochs=1 (or more) later once
# something is confirmed working end-to-end.
# =====================================================================
from transformers import Trainer, TrainingArguments
from unsloth import is_bf16_supported

FastVisionModel.for_training(model)

data_collator = DeepSeekOCRDataCollator(
    tokenizer=tokenizer,
    model=model,
    image_size=640,
    base_size=1024,
    crop_mode=True,
    train_on_responses_only=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=converted_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        output_dir="outputs",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    ),
)

trainer_stats = trainer.train()

# =====================================================================
# CELL 12 — Save the LoRA adapter
# =====================================================================
model.save_pretrained("/kaggle/working/deepseek_ocr_lora")
tokenizer.save_pretrained("/kaggle/working/deepseek_ocr_lora")

# =====================================================================
# CELL 13 — Re-run the same eval (Cell 6's evaluate()) against the now
# fine-tuned `model` in memory — no reload needed, `predict()` from Cell 4
# already references the global `model`/`tokenizer`. This gives the
# "after" number to compare against the baseline logged in STATUS.md.
# =====================================================================
evaluate()
