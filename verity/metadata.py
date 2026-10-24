"""Clue 1: Metadata (EXIF) analysis.

Every camera photo stores hidden details: camera model, capture time,
software used. Editing apps often change or remove these details.
Missing metadata is NOT proof of editing (screenshots and WhatsApp
images also have none), so we only raise "flags" with a weak score.
"""
import re
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

HEAVY_EDITING_SOFTWARE = [
    "photoshop", "gimp", "snapseed", "canva",
    "picsart", "pixlr", "facetune", "remini", "affinity"
]
ROUTINE_EDITING_SOFTWARE = ["lightroom"]

AI_REGEXES = [
    re.compile(r"\bdall[-_\s]?e(?:[-_\s]?\d+)?\b", re.IGNORECASE),
    re.compile(r"\bmidjourney\b", re.IGNORECASE),
    re.compile(r"\bstable[\s_-]?diffusion\b", re.IGNORECASE),
    re.compile(r"\bfirefly\b", re.IGNORECASE),
    re.compile(r"\bcomfyui\b", re.IGNORECASE),
    re.compile(r"\bautomatic1111\b", re.IGNORECASE),
    re.compile(r"\bnovelai\b", re.IGNORECASE),
    re.compile(r"\bimagen\b", re.IGNORECASE),
]

TARGET_METADATA_FIELD_KEYWORDS = [
    "software", "imagedescription", "description", "artist",
    "copyright", "author", "comment", "xmp", "iptc"
]


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
    with Image.open(path) as img:
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
        png_info = {}
        for k, v in img.info.items():
            if isinstance(v, str) and len(v) < 500:
                data[f"info:{k}"] = v
                png_info[k] = v
            elif isinstance(v, (dict, list)):
                png_info[k] = str(v)

    flags, score = [], 0.0

    if not exif:
        flags.append("No camera metadata found. This is common for screenshots, "
                     "social-media downloads, edited or AI-generated images, "
                     "so it is a weak hint, not proof.")
        score += 0.2

    # Check for AI generator references:
    # 1. Target metadata fields: Software, ImageDescription, Artist/Copyright-style, XMP/IPTC
    # 2. PNG info keys and generator metadata keys (e.g. parameters, prompt, workflow)
    ai_found = False
    for k, v in data.items():
        k_lower = k.lower()
        if any(kw in k_lower for kw in TARGET_METADATA_FIELD_KEYWORDS):
            val_str = str(v)
            if any(p.search(val_str) for p in AI_REGEXES):
                ai_found = True
                break

    if not ai_found:
        for k, v in png_info.items():
            k_lower = str(k).lower()
            if "parameters" in k_lower or any(p.search(k_lower) for p in AI_REGEXES):
                ai_found = True
                break
            if k_lower in ("prompt", "workflow", "negative_prompt") or any(p.search(str(v)) for p in AI_REGEXES):
                ai_found = True
                break

    if ai_found:
        flags.append("Metadata mentions an AI image generator.")
        score += 0.8

    # Software edits: distinguish heavy editors (Photoshop) from routine exports (Lightroom)
    # Adobe often names Lightroom as 'Adobe Photoshop Lightroom Classic', so check Lightroom first
    software = str(data.get("Software", ""))
    software_lower = software.lower()
    if any(s in software_lower for s in ROUTINE_EDITING_SOFTWARE):
        flags.append(f"Photo processing/export software recorded: {software} (routine adjustment).")
        score += 0.1
    elif any(s in software_lower for s in HEAVY_EDITING_SOFTWARE):
        flags.append(f"Editing software recorded: {software}")
        score += 0.5

    if exif and "Make" not in data and "Model" not in data:
        flags.append("No camera make/model recorded.")
        score += 0.1

    # Date check: treat DateTime vs DateTimeOriginal difference as weak
    original = data.get("DateTimeOriginal")
    modified = data.get("DateTime")
    if original and modified and str(original) != str(modified):
        diff_sec = None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt_orig = datetime.strptime(str(original).strip(), fmt)
                dt_mod = datetime.strptime(str(modified).strip(), fmt)
                diff_sec = abs((dt_mod - dt_orig).total_seconds())
                break
            except Exception:
                pass

        if diff_sec is not None and diff_sec <= 60:
            flags.append(f"Minor timestamp difference between capture and save "
                         f"({int(diff_sec)}s difference; common after rotate or save).")
            score += 0.05
        else:
            flags.append(f"File saved after capture (taken {original}, saved {modified}).")
            score += 0.1

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
