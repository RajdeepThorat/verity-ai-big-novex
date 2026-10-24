import os
import tempfile
import pytest
from PIL import Image
from verity.metadata import analyze_metadata


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d


def create_jpeg_with_exif(path, exif_tags):
    im = Image.new("RGB", (120, 120), color=(200, 200, 200))
    exif = im.getexif()
    for tag, val in exif_tags.items():
        exif[tag] = val
    im.save(path, exif=exif)


def test_metadata_dallas_skyline_not_flagged_as_ai(tmp_dir):
    """'Sunset over Dallas skyline' should not trigger the AI generator flag."""
    p = os.path.join(tmp_dir, "dallas.jpg")
    create_jpeg_with_exif(p, {0x010E: "Sunset over Dallas skyline", 0x010F: "Sony", 0x0110: "A7III"})
    res = analyze_metadata(p)
    assert not any("AI" in f for f in res["flags"]), f"Unexpected AI flag: {res['flags']}"
    assert res["score"] == 0.0


@pytest.mark.parametrize("ai_caption", [
    "Generated with DALL-E 3",
    "Created by Midjourney v6",
    "Prompt output from Stable Diffusion XL",
    "Processed using Adobe Firefly",
    "ComfyUI generation node",
])
def test_metadata_ai_generators_flagged(tmp_dir, ai_caption):
    """Genuine AI generator mentions in target fields should be flagged."""
    p = os.path.join(tmp_dir, "ai.jpg")
    create_jpeg_with_exif(p, {0x010E: ai_caption, 0x010F: "Sony", 0x0110: "A7III"})
    res = analyze_metadata(p)
    assert any("AI" in f for f in res["flags"]), f"Expected AI flag for '{ai_caption}', got: {res['flags']}"
    assert res["score"] >= 0.8


def test_metadata_png_parameters_flagged(tmp_dir):
    """PNG with 'parameters' info key (Automatic1111/ComfyUI) should be flagged."""
    p = os.path.join(tmp_dir, "ai.png")
    im = Image.new("RGB", (120, 120), color=(100, 100, 100))
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("parameters", "masterpiece, 8k, detailed portrait")
    im.save(p, pnginfo=info)
    res = analyze_metadata(p)
    assert any("AI" in f for f in res["flags"])
    assert res["score"] >= 0.8


def test_metadata_lightroom_vs_photoshop(tmp_dir):
    """Lightroom exports should not have the same heavy editing weight as Photoshop."""
    p_lr = os.path.join(tmp_dir, "lr.jpg")
    create_jpeg_with_exif(p_lr, {0x0131: "Adobe Photoshop Lightroom Classic 13.2", 0x010F: "Canon", 0x0110: "R5"})
    res_lr = analyze_metadata(p_lr)

    p_ps = os.path.join(tmp_dir, "ps.jpg")
    create_jpeg_with_exif(p_ps, {0x0131: "Adobe Photoshop 2024", 0x010F: "Canon", 0x0110: "R5"})
    res_ps = analyze_metadata(p_ps)

    assert res_lr["score"] < res_ps["score"]
    assert any("routine adjustment" in f for f in res_lr["flags"])
    assert any("Editing software recorded" in f for f in res_ps["flags"])


def test_metadata_datetime_difference_weak(tmp_dir):
    """Small seconds-level DateTime differences should be treated as weak."""
    p = os.path.join(tmp_dir, "datetime.jpg")
    create_jpeg_with_exif(p, {
        0x010F: "Nikon",
        0x0110: "Z6",
        0x9003: "2026:05:01 14:30:00",  # DateTimeOriginal
        0x0132: "2026:05:01 14:30:08",  # DateTime (8 seconds later)
    })
    res = analyze_metadata(p)
    # The datetime difference should be weak (0.05)
    assert res["score"] <= 0.1
    assert any("Minor timestamp difference" in f for f in res["flags"])
