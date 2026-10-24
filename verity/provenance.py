"""Clue 7: Provenance and Content Credentials (C2PA / IPTC-XMP) Analysis.

Inspects uploaded media for:
1. C2PA Content Credentials (JUMBF metadata, cryptographically signed manifests,
   claim generator, signers, action assertions like generative AI creation or editing).
2. IPTC/XMP DigitalSourceType declarations (e.g., trainedAlgorithmicMedia,
   compositeWithTrainedAlgorithmicMedia, digitalCapture).

Scoring semantics:
- Valid credentials / XMP declaring AI generation: score >= 0.9 (decisive signal).
- Valid credentials from camera / capture device: score 0.0 (provenance verified).
- Credentials present but signature INVALID or edited after signing: score ~0.7.
- No credentials / no XMP: available=True, score 0.0, weight_factor=0.0 (neutral, no penalty).
- c2pa library not available: graceful fallback with available=False.
"""

import os
import io
import re
import json
import html
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Optional C2PA library import
try:
    import c2pa
    C2PA_AVAILABLE = True
except Exception as _c2pa_err:
    c2pa = None
    C2PA_AVAILABLE = False
    logger.debug("c2pa library not available: %s", _c2pa_err)

AI_SOURCE_TYPES = {
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicallyenhanced",
}

AI_TOOL_KEYWORDS = [
    "firefly", "midjourney", "dall-e", "dalle", "stable diffusion",
    "stablediffusion", "comfyui", "imagen", "novelai", "generator",
    "generative", "bing image creator", "chatgpt", "adobe firefly"
]

CAMERA_SOURCE_TYPES = {
    "digitalcapture",
    "negativefilm",
    "positivefilm",
}

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".avif": "image/avif",
}


def parse_xmp_digital_source_type(file_path: str) -> Optional[str]:
    """Lightweight extraction of IPTC/XMP DigitalSourceType without external C libraries."""
    try:
        with open(file_path, "rb") as f:
            # Read first 5MB which contains headers and metadata segments
            raw = f.read(5 * 1024 * 1024)
    except Exception as e:
        logger.debug("Failed reading file for XMP: %s", e)
        return None

    # Search for XMP packet or xmpmeta block
    m = re.search(rb"<x:xmpmeta [^>]*>.*?</x:xmpmeta>", raw, re.DOTALL)
    if not m:
        m = re.search(rb"<\?xpacket [^>]*\?>.*<\?xpacket [^>]*\?>", raw, re.DOTALL)

    if not m:
        return None

    try:
        xml_str = m.group(0).decode("utf-8", errors="ignore")
        root = ET.fromstring(xml_str)
        for elem in root.iter():
            tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag_name == "DigitalSourceType" and elem.text:
                val = elem.text.strip()
                # If full URI, extract leaf identifier
                return val.rstrip("/").split("/")[-1]
            # Check attribute if inlined
            for attr_name, attr_val in elem.attrib.items():
                if "DigitalSourceType" in attr_name and attr_val:
                    return attr_val.strip().rstrip("/").split("/")[-1]
    except Exception as e:
        logger.debug("Error parsing XMP XML: %s", e)

    # Fallback regex search on raw string
    m_regex = re.search(
        r"(?:DigitalSourceType>|DigitalSourceType=\")([^\s<\"]+)",
        raw.decode("latin-1", errors="ignore")
    )
    if m_regex:
        val = m_regex.group(1).strip()
        return val.rstrip("/").split("/")[-1]

    return None


def _extract_c2pa_info(file_path: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[bool]]:
    """Inspect file for C2PA manifest using c2pa-python if available.

    Returns:
        (has_manifest, manifest_data_dict, signature_valid)
    """
    if not C2PA_AVAILABLE or c2pa is None:
        return False, None, None

    ext = os.path.splitext(file_path)[1].lower()
    mime_type = MIME_MAP.get(ext, "image/jpeg")

    try:
        with open(file_path, "rb") as f:
            reader = c2pa.Reader(mime_type, f)
            manifest_json_str = reader.json()
            if not manifest_json_str:
                return False, None, None

            manifest_data = json.loads(manifest_json_str)

            # Check signature validity
            is_valid = None
            if hasattr(reader, "is_valid"):
                try:
                    is_valid = bool(reader.is_valid) if isinstance(reader.is_valid, bool) else bool(reader.is_valid())
                except Exception:
                    pass

            # Also check validation status in manifest JSON
            active_id = manifest_data.get("active_manifest")
            active_manifest = manifest_data.get("manifests", {}).get(active_id, {})
            val_status = active_manifest.get("validation_status", [])
            val_state = None
            if hasattr(reader, "get_validation_state"):
                try:
                    val_state = reader.get_validation_state()
                except Exception:
                    pass

            # Look for invalid or tampered signals
            has_tamper_flag = False
            if val_status:
                for st in val_status:
                    code = str(st.get("code", "")).lower()
                    explanation = str(st.get("explanation", "")).lower()
                    if any(bad in code or bad in explanation for bad in ["mismatch", "invalid", "tampered", "corrupt", "failed"]):
                        has_tamper_flag = True
                        break

            if val_state and any(bad in str(val_state).lower() for bad in ["invalid", "tampered", "mismatch"]):
                has_tamper_flag = True

            if has_tamper_flag:
                signature_valid = False
            elif is_valid is not None:
                signature_valid = is_valid
            else:
                signature_valid = True if active_id else None

            return True, manifest_data, signature_valid

    except Exception as e:
        # Expected when no JUMBF/C2PA block is found
        logger.debug("C2PA read exception (normal for unsigned media): %s", e)
        return False, None, None


def analyze_provenance(path: str) -> Dict[str, Any]:
    """Analyze C2PA Content Credentials and IPTC/XMP digital source provenance."""
    if not path or not os.path.exists(path):
        return {
            "name": "Provenance",
            "score": 0.0,
            "flags": ["File not found for provenance analysis."],
            "available": False,
            "weight_factor": 0.0,
            "details": {},
        }

    # 1. Parse IPTC/XMP DigitalSourceType
    xmp_source_type = parse_xmp_digital_source_type(path)

    # 2. Parse C2PA credentials
    has_c2pa, c2pa_data, sig_valid = _extract_c2pa_info(path)

    # If neither C2PA library nor XMP is present
    if not C2PA_AVAILABLE and xmp_source_type is None:
        return {
            "name": "Provenance",
            "score": 0.0,
            "flags": ["C2PA Content Credentials verification library is not installed on this system."],
            "available": False,
            "weight_factor": 0.0,
            "details": {"c2pa_library_installed": False},
        }

    claim_generator = None
    signer = None
    actions: List[str] = []
    c2pa_source_types: List[str] = []
    is_ai_action = False
    is_camera_action = False

    if has_c2pa and c2pa_data:
        active_id = c2pa_data.get("active_manifest")
        active_m = c2pa_data.get("manifests", {}).get(active_id, {})
        claim_generator = active_m.get("claim_generator")
        sig_info = active_m.get("signature_info", {})
        signer = sig_info.get("issuer")

        for assertion in active_m.get("assertions", []):
            label = assertion.get("label", "")
            data = assertion.get("data", {})
            if "c2pa.actions" in label and isinstance(data, dict):
                for act in data.get("actions", []):
                    action_name = act.get("action", "")
                    software = act.get("softwareAgent", "")
                    dst = act.get("digitalSourceType", "")
                    if action_name:
                        actions.append(f"{action_name} ({software})" if software else action_name)
                    if dst:
                        leaf_dst = dst.rstrip("/").split("/")[-1]
                        c2pa_source_types.append(leaf_dst)

                    # Check for generative AI indicators
                    if dst and any(ai_term in dst.lower() for ai_term in AI_SOURCE_TYPES):
                        is_ai_action = True
                    if software and any(kw in software.lower() for kw in AI_TOOL_KEYWORDS):
                        is_ai_action = True
                    if any(kw in str(claim_generator).lower() for kw in AI_TOOL_KEYWORDS):
                        is_ai_action = True

                    # Check for camera / capture indicators
                    if dst and any(cam_term in dst.lower() for cam_term in CAMERA_SOURCE_TYPES):
                        is_camera_action = True

    # Check XMP source type as well
    if xmp_source_type:
        st_lower = xmp_source_type.lower()
        if any(ai_term in st_lower for ai_term in AI_SOURCE_TYPES):
            is_ai_action = True
        elif any(cam_term in st_lower for cam_term in CAMERA_SOURCE_TYPES):
            is_camera_action = True

    flags = []
    score = 0.0
    weight_factor = 1.0

    # Decision Logic:
    # 1. Invalid signature or post-signing tampering
    if has_c2pa and sig_valid is False:
        score = 0.70
        flags.append("C2PA Content Credentials present but signature is INVALID or media was modified after signing.")

    # 2. AI declared via C2PA or XMP
    elif is_ai_action:
        score = 0.95
        desc = xmp_source_type or (c2pa_source_types[0] if c2pa_source_types else "generative AI action")
        agent_info = f" via {claim_generator}" if claim_generator else ""
        flags.append(f"Content Credentials / XMP declare AI-generated origin ({desc}{agent_info}).")

    # 3. Valid camera / device capture
    elif (has_c2pa and sig_valid is True and is_camera_action) or (xmp_source_type and is_camera_action and not has_c2pa):
        score = 0.0
        device_info = f" from {signer or claim_generator}" if (signer or claim_generator) else ""
        flags.append(f"Provenance verified: authentic capture from verified hardware camera/device{device_info}.")

    # 4. Valid credentials with routine human edits (e.g. Photoshop / Lightroom)
    elif has_c2pa and sig_valid is True:
        score = 0.15
        flags.append(f"Valid C2PA Content Credentials present ({claim_generator or 'verified signer'}).")

    # 5. No credentials / no XMP
    else:
        score = 0.0
        weight_factor = 0.0  # Must NEVER push verdict or dilute other checks
        flags.append("No C2PA Content Credentials or digital source declarations found (typical for standard photos and social media uploads).")

    summary_parts = []
    if claim_generator:
        summary_parts.append(f"Generator: {claim_generator}")
    if signer:
        summary_parts.append(f"Signer: {signer}")
    if actions:
        summary_parts.append(f"Actions: {', '.join(actions[:3])}")
    if xmp_source_type:
        summary_parts.append(f"XMP DigitalSourceType: {xmp_source_type}")
    if sig_valid is not None:
        summary_parts.append("Signature: " + ("Valid" if sig_valid else "INVALID"))

    details = {
        "has_c2pa": has_c2pa,
        "signature_valid": sig_valid,
        "signer": signer,
        "claim_generator": claim_generator,
        "actions": actions,
        "digital_source_type": xmp_source_type or (c2pa_source_types[0] if c2pa_source_types else None),
        "is_camera_verified": bool(is_camera_action and (sig_valid is not False)),
        "summary": " | ".join(summary_parts) if summary_parts else "No Content Credentials",
    }

    return {
        "name": "Provenance",
        "score": round(score, 3),
        "flags": flags,
        "available": True,
        "weight_factor": weight_factor,
        "details": details,
    }
