"""Run every check on all files in samples/genuine and samples/fake.

Usage:  python test_checks.py
The analyzer deletes the file it analyses (privacy feature), so this
script always works on a temporary COPY of each sample.
"""
import os
import shutil
import tempfile
from verity.analyzer import analyze_file, IMAGE_EXT, VIDEO_EXT

for folder in ["samples/genuine", "samples/fake"]:
    if not os.path.isdir(folder):
        continue
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() not in IMAGE_EXT | VIDEO_EXT:
            continue
        tmp = os.path.join(tempfile.mkdtemp(), name)
        shutil.copy(os.path.join(folder, name), tmp)
        kind, out = analyze_file(tmp)
        if out is None:
            print(f"\n[{folder.split('/')[-1].upper()}] {name} (skipped: could not read)")
            continue
        v = out["verdict"]
        print(f"\n[{folder.split('/')[-1].upper()}] {name}")
        print(f"  RESULT: {v['band']}  (score {v['score']}, confidence {v['confidence']})")
        for r in out["results"]:
            s = f"{r['score']:.2f}" if r.get("available") else "skipped"
            print(f"    - {r['name']:<26} {s:>7}  {r['flags'][0]}")
