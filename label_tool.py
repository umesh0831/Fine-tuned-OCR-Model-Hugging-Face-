#!/usr/bin/env python3
"""
Manual labeling tool for prescription images.
Walks data/train, data/val, data/test, opens each image in Preview, waits
for you to type the transcription in the terminal, and appends it to
ground_truth_manual.csv (with a split column). Resumable: already-labeled
images are skipped on the next run.

Usage:
    python3 label_tool.py
"""
import csv
import os
import subprocess

DATA_DIR = "/Users/rahulroy/Desktop/ocr-finetune-eval/data"
SPLITS = ["train", "val", "test"]
OUTPUT_CSV = "/Users/rahulroy/Desktop/ocr-finetune-eval/data/ground_truth_manual.csv"

FORMAT_HINT = "Name: ... | Age: ... | Date: ... | Rx: ... | Sig: ..."


def load_done():
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["Split"], row["Filename"]))
    return done


def ensure_header():
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="") as fh:
            csv.writer(fh).writerow(["Split", "Filename", "Extracted Text"])


def main():
    ensure_header()
    done = load_done()

    items = []
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        files = sorted(
            (f for f in os.listdir(split_dir) if f.lower().endswith(".jpg")),
            key=lambda f: int(f.split(".")[0]),
        )
        items.extend((split, f) for f in files)

    remaining = [(split, f) for split, f in items if (split, f) not in done]

    print(f"{len(done)} already labeled, {len(remaining)} remaining.\n")
    print(f"Format: {FORMAT_HINT}")
    print("Commands: type transcription + Enter to save | 'skip' to skip | "
          "'quit' to stop\n")

    for split, fname in remaining:
        path = os.path.join(DATA_DIR, split, fname)
        subprocess.run(["open", "-a", "Preview", path])

        text = input(f"[{split}/{fname}] > ").strip()

        if text.lower() == "quit":
            print("Stopping. Progress saved.")
            break
        if text.lower() == "skip":
            continue
        if not text:
            print("Empty input, not saved. Type 'skip' to skip intentionally.")
            continue

        with open(OUTPUT_CSV, "a", newline="") as fh:
            csv.writer(fh).writerow([split, fname, text])
        print(f"Saved ({split}/{fname}).\n")

    print(f"\nDone for now. Labeled file: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
