"""Unit test for VerityAI dataset evaluation harness using a synthetic mini-manifest.

Ensures that eval.py runs completely end-to-end on CI without external downloads:
- Processes real, AI, and edited images plus WhatsApp-style variants
- Evaluates per-check scores and fuse verdicts on temporary copies
- Caches results to scores.csv
- Computes ROC curves, AUC, confusion matrix, and FPR
- Generates markdown report and PNG figures
"""

import os
import csv
import numpy as np
from PIL import Image

from eval import evaluate_manifest, compute_roc_and_auc
from tools.build_dataset import make_whatsapp_variant
from verity.fusion import BAND_CLEAN, BAND_UNSURE, BAND_STRONG


def test_compute_roc_and_auc():
    """Verify empirical ROC curve and AUC calculation."""
    # Perfect separation
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    fpr, tpr, auc = compute_roc_and_auc(y_true, y_scores)
    assert auc == 1.0
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0

    # Random / chance separation
    y_true_rand = np.array([1, 0, 1, 0])
    y_scores_rand = np.array([0.5, 0.5, 0.5, 0.5])
    _, _, auc_rand = compute_roc_and_auc(y_true_rand, y_scores_rand)
    assert 0.4 <= auc_rand <= 0.6


def test_evaluate_mini_manifest(tmp_path):
    """Test full evaluation pipeline on a tiny synthetic dataset."""
    data_dir = tmp_path / "data"
    real_dir = data_dir / "real"
    ai_dir = data_dir / "ai"
    edited_dir = data_dir / "edited"
    real_dir.mkdir(parents=True)
    ai_dir.mkdir(parents=True)
    edited_dir.mkdir(parents=True)

    manifest_rows = []

    # 1. Synthetic Real Images
    for i in range(2):
        im_arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        im = Image.fromarray(im_arr)
        orig_p = real_dir / f"real_{i}.jpg"
        im.save(orig_p, quality=90)

        wa_p = real_dir / f"real_{i}_wa.jpg"
        make_whatsapp_variant(str(orig_p), str(wa_p), quality=70)

        manifest_rows.append({
            "path": str(orig_p),
            "label": "real",
            "group": "original",
            "source": "synthetic",
            "license": "MIT",
            "compressed_variant_of": ""
        })
        manifest_rows.append({
            "path": str(wa_p),
            "label": "real",
            "group": "whatsapp",
            "source": "synthetic",
            "license": "MIT",
            "compressed_variant_of": str(orig_p)
        })

    # 2. Synthetic AI Images (with EXIF tag to trigger AI/metadata detection)
    for i in range(2):
        im = Image.new("RGB", (100, 100), color=(120, 140, 160))
        exif = im.getexif()
        exif[0x010E] = "Generated with Midjourney v6"
        orig_p = ai_dir / f"ai_{i}.jpg"
        im.save(orig_p, exif=exif, quality=90)

        wa_p = ai_dir / f"ai_{i}_wa.jpg"
        make_whatsapp_variant(str(orig_p), str(wa_p), quality=70)

        manifest_rows.append({
            "path": str(orig_p),
            "label": "ai",
            "group": "original",
            "source": "synthetic",
            "license": "MIT",
            "compressed_variant_of": ""
        })
        manifest_rows.append({
            "path": str(wa_p),
            "label": "ai",
            "group": "whatsapp",
            "source": "synthetic",
            "license": "MIT",
            "compressed_variant_of": str(orig_p)
        })

    # 3. Synthetic Edited (Copy-Move) Images
    for i in range(2):
        # Create base with rich texture and a duplicated block
        im_arr = np.random.randint(60, 220, (120, 120, 3), dtype=np.uint8)
        # Duplicate 30x30 block
        im_arr[60:90, 60:90] = im_arr[10:40, 10:40]
        im = Image.fromarray(im_arr)
        orig_p = edited_dir / f"edited_{i}.jpg"
        im.save(orig_p, quality=95)

        wa_p = edited_dir / f"edited_{i}_wa.jpg"
        make_whatsapp_variant(str(orig_p), str(wa_p), quality=70)

        manifest_rows.append({
            "path": str(orig_p),
            "label": "edited",
            "group": "original",
            "source": "synthetic",
            "license": "MIT",
            "compressed_variant_of": ""
        })
        manifest_rows.append({
            "path": str(wa_p),
            "label": "edited",
            "group": "whatsapp",
            "source": "synthetic",
            "license": "MIT",
            "compressed_variant_of": str(orig_p)
        })

    # Write mini-manifest.csv
    manifest_csv = data_dir / "mini_manifest.csv"
    fieldnames = ["path", "label", "group", "source", "license", "compressed_variant_of"]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    scores_csv = tmp_path / "results" / "scores.csv"
    report_md = tmp_path / "results" / "report.md"
    figures_dir = tmp_path / "results" / "figures"

    # Execute evaluate_manifest
    scores = evaluate_manifest(
        manifest_path=str(manifest_csv),
        scores_out=str(scores_csv),
        report_out=str(report_md),
        figures_dir=str(figures_dir),
    )

    # Assertions
    assert len(scores) == 12
    assert os.path.exists(scores_csv)
    assert os.path.exists(report_md)

    # Check generated figures
    assert os.path.exists(figures_dir / "roc_ai_check.png")
    assert os.path.exists(figures_dir / "roc_copymove.png")
    assert os.path.exists(figures_dir / "roc_noise.png")
    assert os.path.exists(figures_dir / "roc_ela.png")
    assert os.path.exists(figures_dir / "roc_overall.png")
    assert os.path.exists(figures_dir / "confusion_matrix.png")

    # Verify scores.csv content
    with open(scores_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 12
        for r in reader:
            assert r["status"] == "ok"
            assert "overall_score" in r
            assert r["predicted_band"] in (BAND_CLEAN, BAND_UNSURE, BAND_STRONG)

    # Verify report.md structure
    with open(report_md, "r", encoding="utf-8") as f:
        content = f.read()
        assert "VerityAI Forensic Evaluation Report" in content
        assert "Executive Summary & Headline Numbers" in content
        assert "Confusion Matrix" in content
        assert "False-Positive Rate Analysis" in content
        assert "Worst Failure Cases Analysis" in content
        assert "Honest Forensic Check Assessment" in content

    # Test cache mechanism on second run
    scores_cached = evaluate_manifest(
        manifest_path=str(manifest_csv),
        scores_out=str(scores_csv),
        report_out=str(report_md),
        figures_dir=str(figures_dir),
    )
    assert len(scores_cached) == 12
