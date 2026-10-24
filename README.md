---
title: VerityAI
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# VerityAI — Explainable Media Authenticity Analyzer

VerityAI analyses an uploaded **image or video** for manipulation indicators and returns an
**evidence-based report** instead of a blind "real/fake" verdict. Built for NOVEX 2026.

## What it checks

| Check | What it looks for |
|---|---|
| Metadata (EXIF) | Editing/AI software tags, missing camera info, date mismatches |
| Error Level Analysis | Regions that recompress differently (possible pasted/edited areas) |
| Noise consistency | Regions whose camera grain differs from the rest |
| Copy-move | Parts of the image duplicated inside the same image |
| AI-generation check | Pretrained model `Organika/sdxl-detector` (artificial vs human) |
| Video: temporal | Sudden jumps between frames (cuts / spliced segments) |
| Video: face region | Face sharpness flicker vs background (possible face blending) |

All signals are combined into three honest bands: **No significant manipulation indicators found**,
**Inconclusive**, or **Strong manipulation indicators found**, with a confidence level and reasons.

## Privacy
- Uploads are deleted immediately after analysis; nothing is stored or used for training.
- GPS location is hidden in the report by default.
- No account or login needed.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py                  # open http://127.0.0.1:7860
```

## Project structure
```
app.py               Gradio web interface
verity/metadata.py   EXIF / metadata checks
verity/ela.py        Error Level Analysis
verity/noise.py      Noise consistency
verity/copymove.py   Copy-move detection (SIFT keypoints)
verity/ai_detect.py  Pretrained AI-image detector
verity/video.py      Frame-level video analysis
verity/fusion.py     Combines signals into a band + confidence
verity/report.py     PDF evidence report
verity/analyzer.py   Runs everything and deletes the upload
```

## Limitations
These are probabilistic indicators, not proof. Heavy compression (e.g. WhatsApp forwards),
very small images, and newer AI generators can weaken or fool individual checks.
