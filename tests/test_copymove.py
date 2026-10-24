import os
import tempfile
import cv2
import numpy as np
import pytest

from verity.copymove import analyze_copymove


def make_real_textured_image(size=512):
    """Generate a realistic textured image with multi-scale natural frequency features."""
    np.random.seed(42)
    img = np.zeros((size, size), dtype=np.float32)
    for freq, amp in [(4, 50), (8, 40), (16, 30), (32, 20), (64, 15), (128, 10)]:
        n = cv2.resize(np.random.randn(freq, freq).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        img += n * amp
    for _ in range(30):
        cx, cy = np.random.randint(30, size - 30, 2)
        r = np.random.randint(6, 25)
        cv2.circle(img, (cx, cy), r, float(np.random.randint(-50, 50)), -1)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def textured_base():
    return make_real_textured_image(512)


def test_copymove_exact_paste(textured_base):
    """Exact translation copy-move should be detected."""
    canvas = textured_base.copy()
    sx, sy, sz = 50, 50, 120
    patch = canvas[sy:sy + sz, sx:sx + sz].copy()
    canvas[260:260 + sz, 260:260 + sz] = patch

    tmp = os.path.join(tempfile.mkdtemp(), "exact.png")
    cv2.imwrite(tmp, canvas)
    res = analyze_copymove(tmp)

    assert res["available"] is True
    assert res["score"] >= 0.5, f"Expected detection score >= 0.5, got {res['score']}"
    assert any("matching point pairs" in f for f in res["flags"])


def test_copymove_rotated_paste(textured_base):
    """15-degree rotated copy-move should be detected with affine RANSAC estimation."""
    canvas = textured_base.copy()
    sx, sy, sz = 50, 50, 120
    patch = canvas[sy:sy + sz, sx:sx + sz].copy()
    M = cv2.getRotationMatrix2D((sz / 2, sz / 2), 15, 1.0)
    rot_patch = cv2.warpAffine(patch, M, (sz, sz), borderMode=cv2.BORDER_REFLECT)
    canvas[260:260 + sz, 260:260 + sz] = rot_patch

    tmp = os.path.join(tempfile.mkdtemp(), "rotated.png")
    cv2.imwrite(tmp, canvas)
    res = analyze_copymove(tmp)

    assert res["available"] is True
    assert res["score"] >= 0.5, f"Expected rotated detection score >= 0.5, got {res['score']}"
    assert any("matching point pairs" in f for f in res["flags"])


def test_copymove_scaled_paste(textured_base):
    """0.8x scaled copy-move should be detected."""
    canvas = textured_base.copy()
    sx, sy, sz = 50, 50, 120
    patch = canvas[sy:sy + sz, sx:sx + sz].copy()
    sc_patch = cv2.resize(patch, None, fx=0.8, fy=0.8)
    ph, pw = sc_patch.shape[:2]
    canvas[260:260 + ph, 260:260 + pw] = sc_patch

    tmp = os.path.join(tempfile.mkdtemp(), "scaled.png")
    cv2.imwrite(tmp, canvas)
    res = analyze_copymove(tmp)

    assert res["available"] is True
    assert res["score"] >= 0.5, f"Expected scaled detection score >= 0.5, got {res['score']}"
    assert any("matching point pairs" in f for f in res["flags"])


def test_copymove_jpeg_recompressed(textured_base):
    """JPEG recompressed (quality 75) copy-move should be detected."""
    canvas = textured_base.copy()
    sx, sy, sz = 50, 50, 120
    patch = canvas[sy:sy + sz, sx:sx + sz].copy()
    buf = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), 75])[1]
    jpg_patch = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    canvas[260:260 + sz, 260:260 + sz] = jpg_patch

    tmp = os.path.join(tempfile.mkdtemp(), "jpeg.png")
    cv2.imwrite(tmp, canvas)
    res = analyze_copymove(tmp)

    assert res["available"] is True
    assert res["score"] >= 0.5, f"Expected JPEG recompressed detection score >= 0.5, got {res['score']}"


def test_copymove_clean_negative(textured_base):
    """Clean image without duplicate regions must return score 0.0."""
    tmp = os.path.join(tempfile.mkdtemp(), "clean.png")
    cv2.imwrite(tmp, textured_base)
    res = analyze_copymove(tmp)

    assert res["available"] is True
    assert res["score"] == 0.0
    assert any("No duplicated regions detected" in f for f in res["flags"])


def test_copymove_repeated_pattern_negative():
    """Repeating patterns (e.g. window/tile grid) should NOT trigger false positive copy-move."""
    tile = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(tile, (5, 5), (35, 35), 255, 2)
    cv2.circle(tile, (20, 20), 8, 200, -1)
    grid = cv2.cvtColor(np.tile(tile, (6, 6)), cv2.COLOR_GRAY2BGR)

    tmp = os.path.join(tempfile.mkdtemp(), "grid.png")
    cv2.imwrite(tmp, grid)
    res = analyze_copymove(tmp)

    assert res["available"] is True
    assert res["score"] == 0.0
    assert any("No duplicated regions detected" in f for f in res["flags"])
