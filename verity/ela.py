"""Clue 2: Error Level Analysis (ELA).

Idea: save the JPEG again at a known quality and compare with the
original. Parts that were already compressed many times change very
little; a pasted or edited region (compressed fewer times) changes more,
so it "glows" in the difference map.

Works best on JPEG photos. On PNG or heavily re-shared images it is weak,
so we lower its weight in those cases.
"""
import io
import numpy as np
import cv2
from PIL import Image, ImageChops


def _blocks(m, block):
    h, w = m.shape
    h, w = h - h % block, w - w % block
    return m[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def _block_anomaly(ela_gray, image_gray, block=16):
    """Score how much a REGION stands out (0 to 0.6).

    Busy textures (trees, towers, text) naturally have high error, so we
    divide the error by the amount of detail in each block. Then we smooth
    over neighbouring blocks, because a real edit covers an area, not a
    single block. ELA is supporting evidence, so its score is capped at 0.6.
    """
    if min(ela_gray.shape) < block * 4:
        return 0.0
    detail = np.abs(cv2.Laplacian(image_gray, cv2.CV_32F))
    ratio = _blocks(ela_gray, block) / (_blocks(detail, block) + 1.0)
    v = cv2.blur(np.log(ratio + 1e-3).astype(np.float32), (3, 3))
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    frac = ((v - med) / mad > 3.5).mean()
    if frac < 0.01:
        return 0.0
    if frac > 0.35:            # half the image "stands out" = global effect, not an edit
        return 0.15
    return float(min(0.6, 0.15 + frac * 3))


def analyze_ela(path, quality=90):
    original = Image.open(path).convert("RGB")
    is_jpeg = Image.open(path).format == "JPEG"

    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue()))

    diff = np.asarray(ImageChops.difference(original, resaved)).astype(np.float32)
    gray = diff.max(axis=2)

    image_gray = np.asarray(original.convert("L")).astype(np.float32)
    score = _block_anomaly(gray, image_gray)

    # Make a visible heatmap: stretch the small differences so humans can see them.
    # Bright = high error level. Look for a REGION that glows differently.
    scale = 255.0 / max(np.percentile(gray, 99.5), 1.0)
    vis = np.clip(gray * scale, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    overlay = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    flags = []
    if not is_jpeg:
        flags.append("Image is not a JPEG, so ELA is less reliable here.")
    if score >= 0.4:
        flags.append("A region recompresses differently from similar areas "
                     "(possible pasted or edited area). Inspect the heatmap.")
    elif score > 0:
        flags.append("Minor error-level differences; often caused by edges or textures.")
    else:
        flags.append("No region stands out in error level.")
    flags.append("ELA is a visual aid: look for one area that glows unlike similar areas.")

    return {
        "name": "Error Level Analysis",
        "score": score,
        "weight_factor": 1.0 if is_jpeg else 0.4,
        "flags": flags,
        "heatmap": overlay,
        "available": True,
    }
