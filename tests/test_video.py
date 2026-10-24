import os
import tempfile
import cv2
import numpy as np
import pytest
from unittest.mock import patch

from verity.video import analyze_video
from verity.fusion import fuse, BAND_STRONG, BAND_CLEAN


@pytest.fixture
def synthetic_multiscene_video():
    """Create a temporary synthetic 6-scene video with 5 hard cuts."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "multiscene.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 10.0, (160, 120))
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for c in colors:
        frame = np.full((120, 160, 3), c, dtype=np.uint8)
        for _ in range(10):
            out.write(frame)
    out.release()
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


@pytest.fixture
def synthetic_smooth_video():
    """Create a temporary synthetic smooth video with no cuts."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "smooth.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 10.0, (160, 120))
    for i in range(60):
        # subtle smooth change
        v = int(100 + 50 * np.sin(i / 10.0))
        frame = np.full((120, 160, 3), (v, v, v), dtype=np.uint8)
        out.write(frame)
    out.release()
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


def test_multiscene_cuts_informational_not_strong(synthetic_multiscene_video):
    """Clean multi-scene video with cuts must NOT get a Strong manipulation verdict."""
    with patch("verity.video.ai_probability", return_value=None):
        out = analyze_video(synthetic_multiscene_video)
    assert out is not None

    tc_res = next(r for r in out["results"] if r["name"] == "Temporal Consistency")
    assert tc_res["score"] <= 0.2, f"Expected capped score, got {tc_res['score']}"
    assert any("edit point" in f for f in tc_res["flags"])
    assert "5 edit point(s) detected" in tc_res["flags"][0]

    verdict = fuse(out["results"])
    assert verdict["band"] != BAND_STRONG, f"Verdict should not be Strong: {verdict}"
    assert verdict["band"] == BAND_CLEAN


def test_smooth_video_no_cuts(synthetic_smooth_video):
    """Smooth video should have 0 edit points detected."""
    with patch("verity.video.ai_probability", return_value=None):
        out = analyze_video(synthetic_smooth_video)
    assert out is not None

    tc_res = next(r for r in out["results"] if r["name"] == "Temporal Consistency")
    assert tc_res["score"] == 0.0
    assert any("no edit points detected" in f for f in tc_res["flags"])
