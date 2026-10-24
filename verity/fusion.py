"""The "detective" step: combine all clues into one honest result.

Each clue gives a suspicion score from 0 (looks fine) to 1 (very
suspicious). We take a weighted average, check whether the clues agree,
and place the result in one of three bands. We NEVER say "100% real".
"""

WEIGHTS = {
    "Metadata": 0.15,
    "Error Level Analysis": 0.25,
    "Noise Consistency": 0.15,
    "Copy-Move": 0.15,
    "AI-Generation Check": 0.30,
    "Temporal Consistency": 0.05,
    "Face Region Consistency": 0.20,
}

BAND_CLEAN = "No significant manipulation indicators found"
BAND_UNSURE = "Inconclusive"
BAND_STRONG = "Strong manipulation indicators found"


def _metadata_score(results):
    for r in results:
        if r["name"] == "Metadata":
            return r["score"]
    return 0.0


def fuse(results, quality_notes=None):
    usable = [r for r in results if r.get("available")]
    if not usable:
        return {"band": BAND_UNSURE, "score": 0.0, "confidence": "Low",
                "reasons": ["No checks could run on this file."]}

    total_w, total = 0.0, 0.0
    for r in usable:
        w = WEIGHTS.get(r["name"], 0.1) * r.get("weight_factor", 1.0)
        total += w * r["score"]
        total_w += w
    combined = total / total_w

    # Metadata, ELA, noise, and temporal cuts are SUPPORTING / INFORMATIONAL clues.
    # Only these checks can count as a decisive strong signal:
    decisive = {"Copy-Move": 0.8, "AI-Generation Check": 0.85,
                "Face Region Consistency": 0.8}
    strong = [r["name"] for r in usable
              if r["name"] in decisive and r["score"] >= decisive[r["name"]]]
    scores = [r["score"] for r in usable]
    disagreement = max(scores) - min(scores)

    # Decide the band. One decisive signal is enough: a fully AI-generated
    # image or a clear copy-paste may leave the other checks quiet.
    if combined >= 0.55 or strong:
        band = BAND_STRONG
    elif (combined >= 0.3 or _metadata_score(usable) >= 0.5
          or any(r["name"] in decisive and r["score"] >= 0.5 for r in usable)):
        band = BAND_UNSURE
    else:
        band = BAND_CLEAN

    # Decide how confident we are in that band
    reasons = []
    if len(usable) <= 2:
        confidence = "Low"
        reasons.append("Only a few checks could run.")
    elif disagreement > 0.6 and band != BAND_CLEAN:
        confidence = "Medium"
        reasons.append("The checks disagree with each other.")
    else:
        confidence = "High" if len(usable) >= 4 else "Medium"
    if quality_notes:
        reasons.extend(quality_notes)
        if confidence == "High":
            confidence = "Medium"
    if strong:
        reasons.append("Strong signal from: " + ", ".join(strong) + ".")
    if _metadata_score(usable) >= 0.5:
        reasons.append("Metadata shows editing or AI software was used on this file.")

    return {"band": band, "score": round(combined, 3), "confidence": confidence,
            "reasons": reasons}


def quality_notes_for(width, height, is_jpeg=True):
    notes = []
    if max(width, height) < 500:
        notes.append("Low resolution weakens forensic checks.")
    if not is_jpeg:
        notes.append("Non-JPEG file: compression-based checks are less reliable.")
    return notes
