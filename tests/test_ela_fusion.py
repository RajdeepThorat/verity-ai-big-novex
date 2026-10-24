import numpy as np
from PIL import Image, ImageDraw
import tempfile, os

from verity.ela import analyze_ela
from verity.fusion import fuse, BAND_CLEAN, BAND_UNSURE, BAND_STRONG


def test_ela_alone_never_pushes_above_clean():
    """High ELA score alone must never push fusion verdict above Clean."""
    results = [
        {"name": "Metadata", "score": 0.0, "available": True, "flags": ["Clean"]},
        {"name": "Error Level Analysis", "score": 0.6, "available": True, "flags": ["A region stands out"]},
        {"name": "Noise Consistency", "score": 0.0, "available": True, "flags": ["Clean"]},
        {"name": "Copy-Move", "score": 0.0, "available": True, "flags": ["Clean"]},
        {"name": "AI-Generation Check", "score": 0.0, "available": True, "flags": ["Clean"]},
    ]
    verdict = fuse(results)
    assert verdict["band"] == BAND_CLEAN, f"Expected {BAND_CLEAN}, got {verdict['band']}"


def test_ela_heatmap_generated():
    """ELA should always produce a valid heatmap image array."""
    tmpdir = tempfile.mkdtemp()
    p = os.path.join(tmpdir, "test.png")
    im = Image.new("RGB", (100, 100), color="white")
    draw = ImageDraw.Draw(im)
    draw.rectangle([20, 20, 60, 60], fill="black")
    im.save(p)

    res = analyze_ela(p)
    assert res["available"] is True
    assert "heatmap" in res
    assert isinstance(res["heatmap"], np.ndarray)
    assert res["heatmap"].shape[0] == 100
    assert res["heatmap"].shape[1] == 100


def test_fusion_bands_remain_consistent():
    """Verify Inconclusive and Strong bands still trigger when genuine indicators exist."""
    # Strong signal (e.g. AI-detect)
    results_strong = [
        {"name": "Metadata", "score": 0.1, "available": True, "flags": ["No make/model"]},
        {"name": "Error Level Analysis", "score": 0.2, "available": True, "flags": []},
        {"name": "Noise Consistency", "score": 0.1, "available": True, "flags": []},
        {"name": "Copy-Move", "score": 0.0, "available": True, "flags": []},
        {"name": "AI-Generation Check", "score": 0.90, "available": True, "flags": ["Strong AI"]},
    ]
    v_strong = fuse(results_strong)
    assert v_strong["band"] == BAND_STRONG

    # Inconclusive signal (e.g. Photoshop editing tag in metadata)
    results_unsure = [
        {"name": "Metadata", "score": 0.5, "available": True, "flags": ["Editing software: Photoshop"]},
        {"name": "Error Level Analysis", "score": 0.2, "available": True, "flags": []},
        {"name": "Noise Consistency", "score": 0.1, "available": True, "flags": []},
        {"name": "Copy-Move", "score": 0.0, "available": True, "flags": []},
        {"name": "AI-Generation Check", "score": 0.1, "available": True, "flags": []},
    ]
    v_unsure = fuse(results_unsure)
    assert v_unsure["band"] == BAND_UNSURE
