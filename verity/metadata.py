"""Clue 1: Metadata (EXIF) analysis.

Every camera photo stores hidden details: camera model, capture time,
software used. Editing apps often change or remove these details.
Missing metadata is NOT proof of editing (screenshots and WhatsApp
images also have none), so we only raise "flags" with a weak score.
"""
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

EDITING_SOFTWARE = ["photoshop", "gimp", "snapseed", "canva", "lightroom",
                    "picsart", "pixlr", "facetune", "remini", "affinity"]
AI_SOFTWARE = ["stable diffusion", "midjourney", "dall", "firefly",
               "comfyui", "automatic1111", "novelai", "imagen"]


def _gps_to_text(gps):
    """Convert the GPS block into readable latitude/longitude."""
    try:
        def to_deg(v):
            d, m, s = [float(x) for x in v]
            return d + m / 60 + s / 3600
        lat = to_deg(gps["GPSLatitude"])
        lon = to_deg(gps["GPSLongitude"])
        if gps.get("GPSLatitudeRef") == "S":
            lat = -lat
        if gps.get("GPSLongitudeRef") == "W":
            lon = -lon
        return f"{lat:.5f}, {lon:.5f}"
    except Exception:
        return "present (unreadable)"


def analyze_metadata(path):
    """Return a dict: score (0-1 suspicion), flags, data, has_gps."""
    img = Image.open(path)
    exif = img.getexif()

    data = {}
    for k, v in exif.items():
        data[str(TAGS.get(k, k))] = v
    try:
        for k, v in exif.get_ifd(0x8769).items():   # detailed camera tags
            data[str(TAGS.get(k, k))] = v
    except Exception:
        pass

    gps_text = None
    try:
        gps_raw = exif.get_ifd(0x8825)                # GPS tags
        if gps_raw:
            gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            gps_text = _gps_to_text(gps)
    except Exception:
        pass

    # PNG files (often from AI tools) keep text info in img.info
    for k, v in img.info.items():
        if isinstance(v, str) and len(v) < 500:
            data[f"info:{k}"] = v

    flags, score = [], 0.0

    if not exif:
        flags.append("No camera metadata found. This is common for screenshots, "
                     "social-media downloads, edited or AI-generated images, "
                     "so it is a weak hint, not proof.")
        score += 0.2

    all_text = " ".join(str(v).lower() for v in data.values())
    software = str(data.get("Software", ""))
    if any(s in all_text for s in AI_SOFTWARE) or "parameters" in str(img.info.keys()).lower():
        flags.append("Metadata mentions an AI image generator.")
        score += 0.8
    if any(s in software.lower() for s in EDITING_SOFTWARE):
        flags.append(f"Editing software recorded: {software}")
        score += 0.5

    if exif and "Make" not in data and "Model" not in data:
        flags.append("No camera make/model recorded.")
        score += 0.1

    original = data.get("DateTimeOriginal")
    modified = data.get("DateTime")
    if original and modified and str(original) != str(modified):
        flags.append(f"File saved after capture (taken {original}, saved {modified}).")
        score += 0.3

    # Keep only short, readable values for the report
    clean = {k: str(v)[:80] for k, v in data.items()
             if not isinstance(v, bytes) and k not in ("MakerNote", "UserComment")}
    if gps_text:
        clean["GPS Location"] = gps_text

    return {
        "name": "Metadata",
        "score": min(score, 1.0),
        "flags": flags or ["No suspicious metadata found."],
        "data": clean,
        "has_gps": gps_text is not None,
        "available": True,
    }
