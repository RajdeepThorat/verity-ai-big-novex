"""Builds the downloadable PDF evidence report (privacy-safe)."""
import os
import tempfile
from datetime import datetime
import numpy as np
from PIL import Image
from fpdf import FPDF

BAND_COLORS = {
    "No significant manipulation indicators found": (22, 163, 74),
    "Inconclusive": (217, 119, 6),
    "Strong manipulation indicators found": (220, 38, 38),
}
DISCLAIMER = ("VerityAI reports probabilistic indicators, not proof. A clean result does "
              "not guarantee authenticity, and a suspicious result does not prove "
              "manipulation. Use this report to support human judgment.")


def _safe(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _add_image(pdf, arr, tmpdir, name, w=170):
    if arr is None:
        return
    p = os.path.join(tmpdir, name + ".jpg")
    img = Image.fromarray(np.asarray(arr).astype(np.uint8))
    img.thumbnail((1200, 1200))
    img.save(p, quality=85)
    h = w * img.height / img.width
    if pdf.get_y() + h > 270:
        pdf.add_page()
    pdf.image(p, x=20, w=w)
    pdf.ln(3)


def build_report(file_name, kind, verdict, results, metadata, mask_gps,
                 images=None):
    """images: list of (title, numpy_array). Returns path of the PDF."""
    tmpdir = tempfile.mkdtemp(prefix="verity_report_")
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "VerityAI Evidence Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, _safe(f"File: {file_name}   |   Type: {kind}   |   "
                         f"Generated: {datetime.now():%d %b %Y, %H:%M}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    r, g, b = BAND_COLORS.get(verdict["band"], (80, 80, 80))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 11, _safe("  Result: " + verdict["band"]), fill=True,
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, _safe(f"Combined suspicion score: {verdict['score']:.2f} / 1.00   |   "
                         f"Confidence in this result: {verdict['confidence']}"),
             new_x="LMARGIN", new_y="NEXT")
    for reason in verdict["reasons"]:
        pdf.multi_cell(0, 5, _safe("- " + reason), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Findings by check", new_x="LMARGIN", new_y="NEXT")
    for res in results:
        pdf.set_font("Helvetica", "B", 10)
        status = f"{res['score']:.2f}" if res.get("available") else "skipped"
        pdf.cell(0, 6, _safe(f"{res['name']}  (suspicion: {status})"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for f in res["flags"]:
            pdf.multi_cell(0, 5, _safe("   - " + f), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if metadata:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Metadata", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        for k, v in list(metadata.items())[:30]:
            if mask_gps and ("GPS" in k):
                v = "[hidden for privacy]"
            pdf.multi_cell(0, 4.5, _safe(f"{k}: {v}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    for i, (title, arr) in enumerate(images or []):
        pdf.set_font("Helvetica", "B", 11)
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.cell(0, 7, _safe(title), new_x="LMARGIN", new_y="NEXT")
        _add_image(pdf, arr, tmpdir, f"img{i}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4.5, _safe(DISCLAIMER + " Your uploaded file was deleted after "
                                 "analysis and was not stored or used for training."))

    out = os.path.join(tmpdir, "VerityAI_Report.pdf")
    pdf.output(out)
    return out
