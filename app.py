"""VerityAI web app (Gradio). Run locally with:  python app.py"""
import html
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
.bar {height:8px; background:rgba(255,255,255,.3); border-radius:99px; margin:8px 0 10px 0;}
.bar div {height:100%; background:white; border-radius:99px;}
.empty {border:1px dashed #666; border-radius:14px; padding:40px; text-align:center; opacity:.7;}

.findings {width:100%; border-collapse:collapse; margin-top:12px;}
.findings th, .findings td {border:1px solid rgba(128,128,128,.4); padding:8px 10px;
                            text-align:left; vertical-align:top;}
.findings th:nth-child(2) {width:100px;}
.badge {display:inline-block; min-width:48px; text-align:center; padding:2px 8px;
        border-radius:99px; color:white; font-weight:600; font-size:13px;}
footer {visibility:hidden}
"""

INTRO = """
# VerityAI — Media Authenticity Analyzer
Upload an **image or video**. VerityAI runs several forensic checks and gives you an
**evidence report**, not a blind verdict.

**Privacy:** your file is kept only temporarily and is cleared automatically from the
server cache within about 90 minutes. Nothing is used for training.
"""

EMPTY = ("<div class='empty'>Upload a file and press <b>Analyze</b> "
         "to see the verdict here.</div>")


def verdict_html(v):
    color, icon = BAND_STYLE.get(v["band"], ("#555", "•"))
    reasons = "".join(f"<p>• {html.escape(str(r))}</p>" for r in v["reasons"])
    pct = max(0, min(100, v["score"] * 100))
    bar = f"<div class='bar'><div style='width:{pct:.0f}%'></div></div>"
    return (f"<div class='verdict' style='background:{color}'>"
            f"<h2>{icon} {html.escape(v['band'])}</h2>"
            f"<p>Combined suspicion score: <b>{v['score']:.2f}</b> / 1.00 &nbsp;|&nbsp; "
            f"Confidence: <b>{html.escape(str(v['confidence']))}</b></p>{bar}{reasons}"
            f"<p style='font-size:12px;margin-top:8px'>These are probabilistic indicators, "
            f"not proof of authenticity or manipulation.</p></div>")


def badge(score):
    if score < 0.3:
        color = "#16a34a"
    elif score <= 0.6:
        color = "#d97706"
    else:
        color = "#dc2626"
    return f"<span class='badge' style='background:{color}'>{score:.2f}</span>"


def findings_html(results):
    rows = []
    for r in results:
        if r.get("available"):
            s = badge(r["score"])
        else:
            s = "<span style='opacity:.6'>skipped</span>"
        flags = html.escape(" ".join(r["flags"]))
        rows.append(f"<tr><td>{html.escape(r['name'])}</td><td>{s}</td><td>{flags}</td></tr>")
    return ("<table class='findings'><thead><tr><th>Check</th><th>Suspicion</th>"
            "<th>What we found</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


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
    if ext in IMAGE_EXT:
        gallery.insert(0, (path, "Original"))   # original first, for side-by-side comparison

    meta = dict(out["metadata"])
    if mask_gps:
        meta = {k: ("[hidden for privacy]" if "GPS" in k else v) for k, v in meta.items()}
    if not meta:
        meta = {"info": "No metadata found in this file."}
    return verdict_html(out["verdict"]), findings_html(out["results"]), gallery, meta, out["report"]


def preview(file):
    hidden = gr.update(value=None, visible=False)
    if file is None:
        return hidden, hidden
    path = file if isinstance(file, str) else file.name
    ext = "." + path.rsplit(".", 1)[-1].lower()
    if ext in IMAGE_EXT:
        return gr.update(value=path, visible=True), hidden
    if ext in VIDEO_EXT:
        return hidden, gr.update(value=path, visible=True)
    return hidden, hidden


def reset_results(_file):
    # verdict, findings, gallery, meta, report
    return EMPTY, "", [], {}, None


with gr.Blocks(title="VerityAI", delete_cache=(1800, 3600)) as demo:
    gr.Markdown(INTRO)

    with gr.Row(equal_height=False):
        # ---- Left: input + actions ----
        with gr.Column(scale=4, min_width=320):
            file_in = gr.File(label="Upload image or video",
                              file_types=sorted(IMAGE_EXT | VIDEO_EXT))
            img_prev = gr.Image(label="Original preview", type="filepath", height=300,
                                interactive=False, visible=False)
            vid_prev = gr.Video(label="Original preview", height=300,
                                interactive=False, visible=False)
            mask = gr.Checkbox(value=True, label="Hide GPS location in the report")
            btn = gr.Button("Analyze", variant="primary", size="lg")
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

        # ---- Right: results ----
        with gr.Column(scale=7):
            verdict = gr.HTML(EMPTY)
            findings = gr.HTML()

    # ---- Bottom: detailed evidence in tabs ----
    with gr.Tabs() as tabs:
        with gr.Tab("Evidence", id="evidence"):
            gallery = gr.Gallery(label="Original, heatmaps, frames, timeline", columns=3,
                                 height=480, object_fit="contain", preview=True,
                                 show_label=False)
        with gr.Tab("Metadata", id="metadata"):
            meta = gr.JSON(label="Metadata")

    # ---- Events ----
    file_in.change(preview, inputs=file_in, outputs=[img_prev, vid_prev])
    file_in.change(reset_results, inputs=file_in,
                   outputs=[verdict, findings, gallery, meta, report])

    # .then runs even if run() raises, so the button can never get stuck disabled
    (btn.click(lambda: gr.update(value="Analyzing…", interactive=False),
               None, btn, queue=False)
        .then(run, inputs=[file_in, mask],
              outputs=[verdict, findings, gallery, meta, report], concurrency_limit=1)
        .then(lambda: (gr.update(value="Analyze", interactive=True),
                       gr.Tabs(selected="evidence")),
              None, [btn, tabs], queue=False))

    demo.queue(max_size=20, default_concurrency_limit=1)

if __name__ == "__main__":
    preload_model()
    demo.launch(css=CSS)