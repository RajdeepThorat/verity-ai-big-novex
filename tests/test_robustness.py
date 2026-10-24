import os
import tempfile
import cv2
import numpy as np
import pytest
from PIL import Image
import gradio as gr

from verity.analyzer import analyze_file, check_magic_bytes, MAX_IMAGE_SIZE, MAX_VIDEO_SIZE, MAX_IMAGE_PIXELS
from verity.video import analyze_video
from app import run


def test_magic_bytes_validation():
    """Spoofed extensions should be rejected based on magic bytes."""
    tmpdir = tempfile.mkdtemp()
    spoofed = os.path.join(tmpdir, "spoofed.jpg")
    with open(spoofed, "wb") as f:
        f.write(b"NOT_A_REAL_IMAGE_JUST_TEXT_CONTENT")

    assert check_magic_bytes(spoofed) is None
    kind, out = analyze_file(spoofed)
    assert kind is None and out is None


def test_max_image_size_rejected():
    """Images exceeding MAX_IMAGE_SIZE should raise ValueError."""
    tmpdir = tempfile.mkdtemp()
    large_img = os.path.join(tmpdir, "large.jpg")
    # Write valid JPEG header then sparse/large content
    with open(large_img, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * (MAX_IMAGE_SIZE + 1024))

    with pytest.raises(ValueError, match="maximum upload size"):
        analyze_file(large_img)


def test_max_video_duration_rejected():
    """Videos exceeding 60s should be rejected."""
    tmpdir = tempfile.mkdtemp()
    long_vid = os.path.join(tmpdir, "long.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 65 seconds at 10 fps = 650 frames
    out = cv2.VideoWriter(long_vid, fourcc, 10.0, (80, 60))
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    for _ in range(650):
        out.write(frame)
    out.release()

    with pytest.raises(ValueError, match="exceeds maximum allowed 60 seconds"):
        analyze_video(long_vid)


def test_pil_max_image_pixels_guard():
    """Ensure PIL MAX_IMAGE_PIXELS guard is set."""
    assert Image.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS
    assert Image.MAX_IMAGE_PIXELS == 50_000_000


def test_app_run_error_handling():
    """app.run should raise gr.Error on invalid/corrupted files."""
    tmpdir = tempfile.mkdtemp()
    corrupt = os.path.join(tmpdir, "corrupt.jpg")
    with open(corrupt, "wb") as f:
        f.write(b"not an image")

    with pytest.raises(gr.Error):
        run(corrupt, mask_gps=True)
