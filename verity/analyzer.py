import os
import logging
from PIL import Image

from .metadata import analyze_metadata
from .ela import analyze_ela
from .noise import analyze_noise
from .copymove import analyze_copymove
from .ai_detect import analyze_ai
from .video import analyze_video
from .fusion import fuse, quality_notes_for
from .report import build_report

logger = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def secure_delete(path):
    """Delete the uploaded file once analysis is finished."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning("Failed to delete file %s: %s", path, e)


def analyze_image_file(path, mask_gps=True):
    with Image.open(path) as img:
        w, h, fmt = img.width, img.height, img.format

    meta = analyze_metadata(path)
    ela = analyze_ela(path)
    noise = analyze_noise(path)
    cm = analyze_copymove(path)
    ai = analyze_ai(path)
    results = [meta, ela, noise, cm, ai]

    verdict = fuse(results, quality_notes_for(w, h, fmt == "JPEG"))
    images = [("Error Level Analysis heatmap", ela.get("heatmap")),
              ("Noise inconsistency map", noise.get("heatmap"))]
    if cm.get("score", 0) > 0:
        images.append(("Copy-move matches", cm.get("heatmap")))
    report = build_report(os.path.basename(path), "Image", verdict, results,
                          meta["data"], mask_gps, images)
    return {"verdict": verdict, "results": results, "images": images,
            "metadata": meta["data"], "has_gps": meta["has_gps"], "report": report}


def analyze_video_file(path, mask_gps=True):
    out = analyze_video(path)
    if out is None:
        return None
    verdict = fuse(out["results"])
    images = [("Frame-level timeline", out["chart"])]
    images += [(f"Sample frame {i + 1}", f) for i, f in enumerate(out["frames"][:4])]
    report = build_report(os.path.basename(path), "Video", verdict, out["results"],
                          out["metadata"], mask_gps, images)
    return {"verdict": verdict, "results": out["results"], "images": images,
            "frames": out["frames"], "chart": out["chart"],
            "metadata": out["metadata"], "has_gps": False, "report": report,
            "duration": out["duration"]}


def analyze_file(path, mask_gps=True):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in IMAGE_EXT:
            return "image", analyze_image_file(path, mask_gps)
        if ext in VIDEO_EXT:
            return "video", analyze_video_file(path, mask_gps)
        return None, None
    finally:
        secure_delete(path)       # privacy: never keep the original upload
