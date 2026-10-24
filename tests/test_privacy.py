import os
import tempfile
import time
import logging
import numpy as np
import pytest
from PIL import Image

from verity.report import build_report, cleanup_stale_reports
from verity.analyzer import secure_delete, analyze_file


def test_build_report_in_memory_images():
    """Report generation should build images in memory and not create image JPEGs on disk."""
    verdict = {
        "band": "No significant manipulation indicators found",
        "score": 0.1,
        "confidence": "High",
        "reasons": ["All checks clear"],
    }
    results = [
        {"name": "Noise Consistency", "score": 0.1, "available": True, "flags": ["Clean"]}
    ]
    meta = {"Make": "Nikon"}
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    pdf_path = build_report("sample.jpg", "Image", verdict, results, meta, True, [("Heatmap", dummy_img)])
    assert os.path.exists(pdf_path)

    report_dir = os.path.dirname(pdf_path)
    files = os.listdir(report_dir)
    assert files == ["VerityAI_Report.pdf"], f"Found unexpected temporary files: {files}"


def test_cleanup_stale_reports():
    """cleanup_stale_reports should remove verity_report_* folders older than threshold."""
    stale_dir = tempfile.mkdtemp(prefix="verity_report_test_")
    past_time = time.time() - 600
    os.utime(stale_dir, (past_time, past_time))
    assert os.path.exists(stale_dir)

    cleanup_stale_reports(max_age_seconds=300)
    assert not os.path.exists(stale_dir)


def test_secure_delete_logs_oserror(caplog):
    """secure_delete should log a warning instead of silently swallowing OSError."""
    with caplog.at_level(logging.WARNING):
        # Passing an invalid path pattern or simulating failure
        import unittest.mock as mock
        with mock.patch("os.path.exists", return_value=True), mock.patch("os.remove", side_effect=OSError("Access denied")):
            secure_delete("fake_file.jpg")
    assert any("Failed to delete file" in record.message for record in caplog.records)


def test_analyze_file_deletes_upload_on_windows():
    """analyze_file should close all image handles so secure_delete removes the file on Windows."""
    tmp = os.path.join(tempfile.mkdtemp(), "test_upload.png")
    Image.new("RGB", (64, 64), color="blue").save(tmp)
    assert os.path.exists(tmp)

    from unittest.mock import patch
    with patch("verity.ai_detect.ai_probability", return_value=None):
        kind, out = analyze_file(tmp)
    # File must be deleted cleanly without Windows file-locking PermissionError
    assert not os.path.exists(tmp), f"File {tmp} was not deleted after analysis!"
