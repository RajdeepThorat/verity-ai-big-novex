import os
import logging
from PIL import Image

from .metadata import analyze_metadata
from .ela import analyze_ela
from .noise import analyze_noise
from .copymove import analyze_copymove
from .ai_detect import analyze_ai
from .provenance import analyze_provenance
from .video import analyze_video
from .fusion import fuse, quality_notes_for
from .report import build_report

logger = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MAX_IMAGE_SIZE = 20 * 1024 * 1024     # 20 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024     # 50 MB
MAX_IMAGE_PIXELS = 50_000_000         # 50 Megapixels
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def check_magic_bytes(path):
    """Verify that file header matches a known image or video format."""
    try:
        with open(path, "rb") as f:
            header = f.read(32)
    except Exception:
        return None

    if len(header) < 4:
        return None

    # Common image magic bytes
    if header.startswith(b"\xff\xd8\xff"):
        return "image"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image"
    if header.startswith(b"BM"):
        return "image"
    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "image"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image"

    # Common video magic bytes
    if len(header) >= 8 and (header[4:8] in (b"ftyp", b"moov", b"mdat", b"wide")):
        return "video"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"AVI ":
        return "video"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "video"

    return None


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
    prov = analyze_provenance(path)
    results = [meta, ela, noise, cm, ai, prov]

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
    if not path or not os.path.exists(path):
        return None, None

    ext = os.path.splitext(path)[1].lower()
    try:
        file_size = os.path.getsize(path)
        magic_kind = check_magic_bytes(path)
        if magic_kind is None:
            return None, None

        if ext in IMAGE_EXT and magic_kind == "image":
            if file_size > MAX_IMAGE_SIZE:
                raise ValueError(f"Image file exceeds maximum upload size of {MAX_IMAGE_SIZE // (1024 * 1024)} MB.")
            return "image", analyze_image_file(path, mask_gps)

        if ext in VIDEO_EXT and magic_kind == "video":
            if file_size > MAX_VIDEO_SIZE:
                raise ValueError(f"Video file exceeds maximum upload size of {MAX_VIDEO_SIZE // (1024 * 1024)} MB.")
            return "video", analyze_video_file(path, mask_gps)

        return None, None
    finally:
        secure_delete(path)       # privacy: never keep the original upload
