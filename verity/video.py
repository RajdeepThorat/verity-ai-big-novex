"""Video analysis: a video is many images (frames) in a row.

1. Sample frames evenly across the video.
2. Run the AI-generation check on each sampled frame.
3. Temporal consistency: real footage changes smoothly; a sudden jump
   between two frames can mean a cut or a spliced segment.
4. Face region consistency: deepfakes paste a generated face onto a real
   head. The face area's sharpness often flickers compared with the
   background. We track that ratio across frames.
"""
import json
import shutil
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .ai_detect import ai_probability

_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def video_metadata(path):
    """Read container info with ffprobe if it is installed."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True, timeout=20).stdout
        tags = json.loads(out).get("format", {}).get("tags", {})
        return {k: str(v)[:80] for k, v in tags.items()}
    except Exception:
        return {}


def analyze_video(path, n_frames=16):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if total <= 0:
        return None

    # 1) consecutive-frame differences over the whole video (downscaled)
    diffs, prev = [], None
    step = max(1, total // 400)            # at most ~400 checks
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g

    # 2) sampled key frames for AI + face checks
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    frames, ai_scores, face_ratios, times = [], [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        times.append(i / fps)
        ai_scores.append(ai_probability(Image.fromarray(rgb)))

        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = _face.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face_s = _sharpness(gray[y:y + h, x:x + w])
            bg_s = _sharpness(gray) + 1e-6
            face_ratios.append(face_s / bg_s)
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 120), 2)
        else:
            face_ratios.append(None)
    cap.release()

    results = []

    # Temporal consistency score
    d = np.array(diffs)
    if len(d) > 5:
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826 + 1e-6
        spikes = int(np.sum((d - med) / mad > 8))
        t_score = 0.0 if spikes == 0 else min(1.0, 0.25 + spikes * 0.15)
        results.append({"name": "Temporal Consistency", "score": t_score, "available": True,
                        "flags": [f"{spikes} sudden jump(s) between frames (cuts or splices)."
                                  if spikes else "Frames change smoothly over time."]})

    # Face region consistency score
    fr = [r for r in face_ratios if r is not None]
    if len(fr) >= 4:
        cv_ratio = float(np.std(fr) / (np.mean(fr) + 1e-6))
        f_score = float(np.clip((cv_ratio - 0.35) / 0.6, 0, 1))
        results.append({"name": "Face Region Consistency", "score": f_score, "available": True,
                        "flags": ["Face sharpness flickers relative to the background "
                                  "(possible face blending)." if f_score >= 0.4 else
                                  "Face region looks stable across frames."]})
    else:
        results.append({"name": "Face Region Consistency", "score": 0.0, "available": False,
                        "flags": ["Not enough clear faces to check."]})

    # AI check across frames
    valid_ai = [s for s in ai_scores if s is not None]
    if valid_ai:
        mean_ai = float(np.mean(valid_ai))
        high = sum(s >= 0.7 for s in valid_ai)
        results.append({"name": "AI-Generation Check", "score": mean_ai, "available": True,
                        "flags": [f"Average AI-pattern score {mean_ai:.0%}; "
                                  f"{high} of {len(valid_ai)} frames above 70%."]})
    else:
        results.append({"name": "AI-Generation Check", "score": 0.0, "available": False,
                        "flags": ["AI model could not be loaded, so this check was skipped."]})

    chart = _timeline_chart(times, ai_scores, face_ratios, d, total / fps)
    return {"results": results, "frames": frames, "chart": chart,
            "duration": total / fps, "fps": fps, "metadata": video_metadata(path)}


def _timeline_chart(times, ai_scores, face_ratios, diffs, duration):
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
    if len(diffs):
        ax[0].plot(np.linspace(0, duration, len(diffs)), diffs, color="#2563eb", lw=1)
    ax[0].set_ylabel("Frame change")
    ax[0].set_title("Frame-level timeline", fontsize=10)
    if any(s is not None for s in ai_scores):
        ax[1].plot(times, [s if s is not None else np.nan for s in ai_scores],
                   "o-", color="#dc2626", label="AI-pattern score")
    ax[1].plot(times, [r if r is not None else np.nan for r in face_ratios],
               "s--", color="#16a34a", label="Face/background sharpness")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
