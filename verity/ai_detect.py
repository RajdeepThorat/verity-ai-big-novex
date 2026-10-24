"""Clue 5: AI-generation check with a PRETRAINED model.

We do not train a model from scratch (that needs huge data and GPUs).
We use "Organika/sdxl-detector" from Hugging Face, which was already
trained to label images as "artificial" or "human".
It downloads automatically the first time (about 350 MB).

Limits: tuned mainly for diffusion images (like SDXL); may miss newer
generators and can be confused by heavy filters. It is one clue, not proof.
"""
from PIL import Image

MODEL_ID = "Organika/sdxl-detector"
_classifier = None
_load_error = None


def _get_classifier():
    global _classifier, _load_error
    if _classifier is None and _load_error is None:
        try:
            from transformers import pipeline
            _classifier = pipeline("image-classification", model=MODEL_ID)
        except Exception as e:          # no internet / no torch etc.
            _load_error = str(e)
    return _classifier


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
