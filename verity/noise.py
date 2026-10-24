"""Clue 3: Noise consistency.

Every camera sensor adds a fine, even "grain" (noise) to the whole photo.
If a region came from another photo, or was smoothed / AI-filled, its
grain is different. We remove the image content with a median filter,
keep only the leftover noise, and compare noise strength block by block.
"""
import numpy as np
import cv2


def analyze_noise(path, block=32):
    img = cv2.imread(path)
    if img is None:
        return {"name": "Noise Consistency", "available": False, "score": 0.0,
                "flags": ["Could not read image."]}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)

    h, w = residual.shape
    h, w = h - h % block, w - w % block
    if h < block * 4 or w < block * 4:
        return {"name": "Noise Consistency", "available": False, "score": 0.0,
                "flags": ["Image too small for noise analysis."]}

    r = residual[:h, :w].reshape(h // block, block, w // block, block)
    local_std = r.std(axis=(1, 3))

    # ignore flat areas (sky, walls) where noise is naturally tiny
    content = gray[:h, :w].reshape(h // block, block, w // block, block).std(axis=(1, 3))
    valid = content > 3
    if valid.sum() < 10:
        valid = np.ones_like(valid, bool)

    vals = np.log(local_std[valid] + 1e-3)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    z = np.zeros_like(local_std)
    z[valid] = (vals - med) / mad
    frac = (np.abs(z) > 4).mean()

    if frac < 0.01:
        score = 0.0
    elif frac > 0.35:
        score = 0.2
    else:
        score = min(0.6, frac * 4 + 0.15)   # supporting evidence only, capped

    vis = np.clip(np.abs(z) / 6 * 255, 0, 255).astype(np.uint8)
    vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_NEAREST)
    heat = cv2.cvtColor(cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    flags = ["Some areas have a different noise pattern from the rest of the image."
             if score >= 0.4 else "Noise pattern is broadly consistent."]
    return {"name": "Noise Consistency", "score": float(score), "flags": flags,
            "heatmap": heat, "available": True}
