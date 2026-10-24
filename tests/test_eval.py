import os
import tempfile
import csv
import cv2
import numpy as np
from PIL import Image
from unittest.mock import patch

from eval import evaluate
from verity.fusion import BAND_CLEAN, BAND_STRONG


def test_eval_script(tmp_path):
    """Test eval.py generates CSV and confusion matrix for genuine and fake samples."""
    samples_dir = tmp_path / "samples"
    genuine_dir = samples_dir / "genuine"
    fake_dir = samples_dir / "fake"
    genuine_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    # Genuine sample: clean photo
    gen_img = Image.new("RGB", (120, 120), color=(180, 180, 180))
    gen_img.save(genuine_dir / "clean_photo.jpg")

    # Fake sample: image with AI generator tag
    fake_img = Image.new("RGB", (120, 120), color=(100, 100, 100))
    exif = fake_img.getexif()
    exif[0x010E] = "Generated with Midjourney v6"
    fake_img.save(fake_dir / "ai_generated.jpg", exif=exif)

    csv_file = tmp_path / "eval_output.csv"

    with patch("verity.ai_detect.ai_probability", return_value=None):
        rows, confusion = evaluate(str(samples_dir), str(csv_file))

    assert len(rows) == 2
    assert os.path.exists(csv_file)

    # Verify CSV structure
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert "Metadata" in reader[0]
        assert "predicted_band" in reader[0]

    # Check confusion matrix counts
    assert confusion["genuine"][BAND_CLEAN] >= 1
    assert confusion["fake"][BAND_STRONG] >= 1 or confusion["fake"]["Inconclusive"] >= 1
