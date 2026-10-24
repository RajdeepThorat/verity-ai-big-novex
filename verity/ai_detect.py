"""Clue 5: AI-generation check with a PRETRAINED model.

We do not train a model from scratch (that needs huge data and GPUs).
We use "Organika/sdxl-detector" from Hugging Face, which was already
trained to label images as "artificial" or "human".
It downloads automatically the first time (about 350 MB).

Limits: tuned mainly for diffusion images (like SDXL); may miss newer
generators and can be confused by heavy filters. It is one clue, not proof.
"""
import os
import time
import threading
from PIL import Image

MODEL_ID = "Organika/sdxl-detector"
_classifier = None
_load_error = None
_last_attempt_time = 0.0
_backoff_seconds = 5.0
_lock = threading.Lock()


def _configure_torch():
    """Set torch threads sensibly for CPU inference."""
    try:
        import torch
        num_threads = min(4, max(1, os.cpu_count() or 1))
        torch.set_num_threads(num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(min(2, num_threads))
            except RuntimeError:
                pass
    except Exception:
        pass


def _load_pipeline():
    """Configure torch and initialize Hugging Face classification pipeline."""
    _configure_torch()
    from transformers import pipeline
    return pipeline("image-classification", model=MODEL_ID)


def _get_classifier():
    """Load the classifier with retry backoff instead of caching error forever."""
    global _classifier, _load_error, _last_attempt_time, _backoff_seconds
    if _classifier is not None:
        return _classifier

    with _lock:
        if _classifier is not None:
            return _classifier

        now = time.time()
        if _load_error is not None and (now - _last_attempt_time) < _backoff_seconds:
            return None

        _last_attempt_time = now
        try:
            _classifier = _load_pipeline()
            _load_error = None
            _backoff_seconds = 5.0
        except Exception as e:
            _load_error = str(e)
            _backoff_seconds = min(60.0, _backoff_seconds * 2.0)

    return _classifier


def _preload_worker():
    _get_classifier()


def preload_model():
    """Start model loading in a background thread at app startup."""
    thread = threading.Thread(target=_preload_worker, daemon=True, name="VerityAI-ModelPreload")
    thread.start()
    return thread


def ai_probability(pil_image):
    """Return probability (0-1) that the image is AI-generated, or None."""
    clf = _get_classifier()
    if clf is None:
        return None
    results = clf(pil_image.convert("RGB"))
    for r in results:
        if r["label"].lower() == "artificial":
            return float(r["score"])
    for r in results:
        if r["label"].lower() == "human":
            return 1.0 - float(r["score"])
    return None


def analyze_ai(path):
    with Image.open(path) as img:
        prob = ai_probability(img)
    if prob is None:
        return {"name": "AI-Generation Check", "available": False, "score": 0.0,
                "flags": ["AI model could not be loaded, so this check was skipped."]}
    if prob >= 0.8:
        msg = f"Model sees strong AI-generation patterns ({prob:.0%})."
    elif prob >= 0.5:
        msg = f"Model sees some AI-generation patterns ({prob:.0%}). Treat with caution."
    else:
        msg = f"Model sees few AI-generation patterns ({prob:.0%})."
    return {"name": "AI-Generation Check", "score": prob, "flags": [msg],
            "available": True}
