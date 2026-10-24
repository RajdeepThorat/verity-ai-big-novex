"""VerityAI web app (Gradio). Run locally with:  python app.py"""
import gradio as gr
from verity.analyzer import analyze_file, IMAGE_EXT, VIDEO_EXT
from verity.ai_detect import preload_model

BAND_STYLE = {
    "No significant manipulation indicators found": ("#16a34a", "✓"),
    "Inconclusive": ("#d97706", "?"),
    "Strong manipulation indicators found": ("#dc2626", "!"),
}

CSS = """
.verdict {border-radius:14px; padding:18px 22px; color:white;}
.verdict h2 {margin:0 0 6px 0; color:white;}
.verdict p {margin:2px 0; opacity:.95;}
footer {visibility:hidden}
"""

INTRO = """
# VerityAI — Media Authenticity Analyzer
Upload an **image or video**. VerityAI runs several forensic checks and gives you an
**evidence report**, not a blind verdict.

**Privacy:** your file is analysed temporarily and **deleted right after analysis**.
Nothing is stored or used for training.
"""


def verdict_html(v):
    color, icon = BAND_STYLE.get(v["band"], ("#555", "•"))
    reasons = "".join(f"<p>• {r}</p>" for r in v["reasons"])
    return (f"<div class='verdict' style='background:{color}'>"
            f"<h2>{icon} {v['band']}</h2>"
            f"<p>Combined suspicion score: <b>{v['score']:.2f}</b> / 1.00 &nbsp;|&nbsp; "
            f"Confidence: <b>{v['confidence']}</b></p>{reasons}"
            f"<p style='font-size:12px;margin-top:8px'>These are probabilistic indicators, "
            f"not proof of authenticity or manipulation.</p></div>")


import html


def findings_md(results):
    lines = ["| Check | Suspicion | What we found |", "|---|---|---|"]
    for r in results:
        s = f"{r['score']:.2f}" if r.get("available") else "skipped"
        flags_escaped = [html.escape(str(f)) for f in r.get("flags", [])]
        if r.get("name") == "Provenance" and r.get("details"):
            d = r["details"]
            details_lines = []
            if d.get("claim_generator"):
                details_lines.append(f"<b>Generator:</b> {html.escape(str(d['claim_generator']))}")
            if d.get("signer"):
                details_lines.append(f"<b>Signer:</b> {html.escape(str(d['signer']))}")
            if d.get("actions"):
                details_lines.append(f"<b>Actions:</b> {html.escape(', '.join(str(a) for a in d['actions']))}")
            if d.get("signature_valid") is not None:
                val_str = "Valid" if d["signature_valid"] else "<span style='color:red'>INVALID / MODIFIED</span>"
                details_lines.append(f"<b>Validity:</b> {val_str}")
            if details_lines:
                flags_escaped.append("<br>" + " | ".join(details_lines))
        lines.append(f"| {html.escape(r['name'])} | {s} | {' '.join(flags_escaped)} |")
    return "\n".join(lines)


def run(file, mask_gps):
    if file is None:
        raise gr.Error("Please upload an image or video first.")
    path = file if isinstance(file, str) else file.name
    ext = "." + path.rsplit(".", 1)[-1].lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        raise gr.Error("Unsupported file type. Use JPG, PNG, WEBP, MP4, MOV, AVI or WEBM.")

    try:
        kind, out = analyze_file(path, mask_gps=mask_gps)
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Analysis failed: {str(e)}")

    if out is None:
        raise gr.Error("Could not read or verify this file. Please ensure it is a valid, uncorrupted image or video.")

    gallery = [(img, title) for title, img in out["images"] if img is not None]
    meta = dict(out["metadata"])
    if mask_gps:
        meta = {k: ("[hidden for privacy]" if "GPS" in k else v) for k, v in meta.items()}
    if not meta:
        meta = {"info": "No metadata found in this file."}
    return verdict_html(out["verdict"]), findings_md(out["results"]), gallery, meta, out["report"]


with gr.Blocks(title="VerityAI", delete_cache=(1800, 3600)) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Upload image or video",
                              file_types=sorted(IMAGE_EXT | VIDEO_EXT))
            mask = gr.Checkbox(value=True, label="Hide GPS location in the report")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            verdict = gr.HTML()
            findings = gr.Markdown()
    gallery = gr.Gallery(label="Evidence (heatmaps, frames, timeline)", columns=2, height="auto")
    with gr.Row():
        meta = gr.JSON(label="Metadata")
        report = gr.File(label="Download evidence report (PDF)")

    with gr.Accordion("How to read the results", open=False):
        gr.Markdown("""
- **Error Level Analysis:** bright regions recompress differently. Look for one *area* that glows unlike the rest.
- **Noise map:** bright blocks have a different camera-grain pattern.
- **Copy-Move:** lines join regions that look duplicated inside the same image.
- **AI-Generation Check:** a pretrained model (Organika/sdxl-detector) estimates AI patterns.
- **Video:** frames are sampled; sudden jumps and face flicker are tracked over time.
- **Inconclusive** is an honest answer: heavy compression (e.g. WhatsApp forwards) weakens many checks.
""")

    btn.click(run, inputs=[file_in, mask], outputs=[verdict, findings, gallery, meta, report], concurrency_limit=1)
    demo.queue(max_size=20, default_concurrency_limit=1)

if __name__ == "__main__":
    preload_model()
    demo.launch(css=CSS)
