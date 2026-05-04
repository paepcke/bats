#!/usr/bin/env python
# ****************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-03 12:40:19
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-03 12:42:18
# ****************************************************

import os
import shutil
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

SOURCE_ROOT = Path("/data2/marsh")

SELECTED_DIRS = [
    "20000101", "20131102", "20131229", "20140429", "20141103",
    "20150726", "20150727", "20160511", "20161226", "20170101",
    "20170213", "20180604", "20181020", "20190729", "20191121",
    "20200213", "20200228", "20210613", "20211223", "20220325",
    "20220813", "20230724", "20231206", "20240117",
]

BATCH_ROOTS = [
    Path("/data/win_share/batch1/input"),
    Path("/data/win_share/batch2/input"),
    Path("/data/win_share/batch3/input"),
    Path("/data/win_share/batch4/input"),
]

# ── Gather all .wav files ──────────────────────────────────────────────────────

wav_files = []
for d in SELECTED_DIRS:
    subdir = SOURCE_ROOT / d
    if not subdir.is_dir():
        print(f"WARNING: {subdir} not found, skipping")
        continue
    found = sorted(subdir.glob("*.wav"))
    print(f"  {d}: {len(found)} .wav files")
    wav_files.extend(found)

total = len(wav_files)
print(f"\nTotal .wav files: {total}")

# ── Partition into 4 roughly equal sets ───────────────────────────────────────

# Compute slice boundaries so the first (total % 4) batches get one extra file
n = 4
base, extra = divmod(total, n)
batches = []
start = 0
for i in range(n):
    size = base + (1 if i < extra else 0)
    batches.append(wav_files[start : start + size])
    start += size

for i, batch in enumerate(batches):
    print(f"  batch{i+1}: {len(batch)} files")

# ── Create output dirs and copy ────────────────────────────────────────────────

for dest_dir, batch in zip(BATCH_ROOTS, batches):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying {len(batch)} files → {dest_dir}")
    for src in batch:
        shutil.copy2(src, dest_dir / src.name)
    print(f"  Done.")

print("\nAll batches complete.")
