"""Evaluation script for VerityAI.

Runs all forensic checks over samples/genuine and samples/fake,
records per-check scores and overall verdicts, writes an evaluation CSV,
and outputs a confusion matrix with per-check diagnostic statistics.

Usage:
    python eval.py [--samples-dir samples] [--csv-out eval_results.csv]
"""
import os
import csv
import shutil
import tempfile
import argparse
from collections import defaultdict
from verity.analyzer import analyze_file, IMAGE_EXT, VIDEO_EXT
from verity.fusion import BAND_CLEAN, BAND_UNSURE, BAND_STRONG

CHECK_NAMES = [
    "Metadata",
    "Error Level Analysis",
    "Noise Consistency",
    "Copy-Move",
    "AI-Generation Check",
    "Temporal Consistency",
    "Face Region Consistency",
]


def evaluate(samples_dir="samples", csv_out="eval_results.csv"):
    rows = []
    confusion = {
        "genuine": defaultdict(int),
        "fake": defaultdict(int),
    }
    check_scores = {
        "genuine": defaultdict(list),
        "fake": defaultdict(list),
    }

    folders = [("genuine", os.path.join(samples_dir, "genuine")),
               ("fake", os.path.join(samples_dir, "fake"))]

    total_files = 0
    for gt, folder in folders:
        if not os.path.isdir(folder):
            continue

        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXT | VIDEO_EXT:
                continue

            total_files += 1
            src_path = os.path.join(folder, name)
            # Use temporary copy because analyze_file deletes the analyzed file (privacy feature)
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, name)
            shutil.copy(src_path, tmp_path)

            try:
                kind, out = analyze_file(tmp_path)
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
                continue

            if out is None:
                print(f"Skipped {name}: could not parse file")
                continue

            verdict = out["verdict"]
            pred_band = verdict["band"]
            score = verdict["score"]
            conf = verdict["confidence"]
            confusion[gt][pred_band] += 1

            row = {
                "file": name,
                "ground_truth": gt,
                "media_type": kind,
                "predicted_band": pred_band,
                "overall_score": score,
                "confidence": conf,
            }

            results_by_name = {r["name"]: r for r in out.get("results", [])}
            for chk in CHECK_NAMES:
                r = results_by_name.get(chk)
                if r and r.get("available"):
                    chk_score = round(float(r["score"]), 3)
                    check_scores[gt][chk].append(chk_score)
                else:
                    chk_score = ""
                row[chk] = chk_score

            rows.append(row)

    # Write CSV output
    if rows:
        fieldnames = ["file", "ground_truth", "media_type", "predicted_band",
                      "overall_score", "confidence"] + CHECK_NAMES
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[+] Wrote detailed results for {len(rows)} samples to {csv_out}")

    # Output Confusion Matrix and Diagnostics
    print("\n" + "=" * 60)
    print("VERITYAI FORENSIC EVALUATION REPORT")
    print("=" * 60)

    if total_files == 0:
        print("\nNote: samples/genuine and samples/fake directories are currently empty.")
        print("To evaluate, add test images/videos to samples/genuine and samples/fake.")
        return rows, confusion

    print("\nCONFUSION MATRIX:")
    bands = [BAND_CLEAN, BAND_UNSURE, BAND_STRONG]
    header = f"{'Ground Truth':<15} | " + " | ".join(f"{b:<18}" for b in ["Clean", "Inconclusive", "Strong"])
    print(header)
    print("-" * len(header))
    for gt in ["genuine", "fake"]:
        c_clean = confusion[gt][BAND_CLEAN]
        c_unsure = confusion[gt][BAND_UNSURE]
        c_strong = confusion[gt][BAND_STRONG]
        tot = c_clean + c_unsure + c_strong
        p_clean = f"{c_clean} ({c_clean/tot:.0%})" if tot else "0"
        p_unsure = f"{c_unsure} ({c_unsure/tot:.0%})" if tot else "0"
        p_strong = f"{c_strong} ({c_strong/tot:.0%})" if tot else "0"
        print(f"{gt.capitalize():<15} | {p_clean:<18} | {p_unsure:<18} | {p_strong:<18}")

    print("\nPER-CHECK AVERAGE SCORES (Suspicion Level):")
    print(f"{'Check Name':<28} | {'Genuine Avg':<12} | {'Fake Avg':<12}")
    print("-" * 58)
    for chk in CHECK_NAMES:
        g_scores = check_scores["genuine"].get(chk, [])
        f_scores = check_scores["fake"].get(chk, [])
        g_avg = f"{sum(g_scores)/len(g_scores):.3f}" if g_scores else "N/A"
        f_avg = f"{sum(f_scores)/len(f_scores):.3f}" if f_scores else "N/A"
        print(f"{chk:<28} | {g_avg:<12} | {f_avg:<12}")

    return rows, confusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VerityAI on genuine and fake media samples.")
    parser.add_argument("--samples-dir", default="samples", help="Path to samples folder containing genuine/ and fake/")
    parser.add_argument("--csv-out", default="eval_results.csv", help="Output CSV filename")
    args = parser.parse_args()
    evaluate(args.samples_dir, args.csv_out)
