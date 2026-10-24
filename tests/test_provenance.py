"""Tests for verity.provenance: C2PA Content Credentials and IPTC/XMP DigitalSourceType.

Covers:
1. In-test real XMP fixture: AI-declared (trainedAlgorithmicMedia)
2. In-test real XMP fixture: Camera-declared (digitalCapture)
3. In-test real image fixture: No credentials / neutral response
4. C2PA parsing: Valid AI-declared manifest (score >= 0.9, decisive)
5. C2PA parsing: Valid camera-declared manifest (score 0.0, provenance verified)
6. C2PA parsing: Invalid signature / tampered credentials (score ~0.7)
7. C2PA library missing fallback (available=False without error)
8. Fusion behavior: No credentials never penalizes or changes band; camera credentials
   raise confidence on clean files but never blindly overwrite visual manipulation flags.
9. HTML escaping in findings_md and PDF report generation with provenance details.
"""

import os
import html
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image, PngImagePlugin

from verity.provenance import (
    analyze_provenance,
    parse_xmp_digital_source_type,
    C2PA_AVAILABLE,
)
from verity.fusion import fuse, BAND_CLEAN, BAND_STRONG, BAND_UNSURE
from verity.report import build_report
from app import findings_md


def create_xmp_png(tmp_path, digital_source_type: str, filename: str = "test_xmp.png") -> str:
    """Helper to generate a real, self-contained PNG image with an embedded XMP packet."""
    xmp_packet = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">
   <Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/{digital_source_type}</Iptc4xmpExt:DigitalSourceType>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""
    p = tmp_path / filename
    meta = PngImagePlugin.PngInfo()
    meta.add_text("XML:com.adobe.xmp", xmp_packet)
    im = Image.new("RGB", (60, 60), color=(150, 150, 150))
    im.save(str(p), pnginfo=meta)
    return str(p)


def test_xmp_ai_declared(tmp_path):
    """Real fixture: XMP DigitalSourceType declaring trainedAlgorithmicMedia (AI)."""
    img_path = create_xmp_png(tmp_path, "trainedAlgorithmicMedia", "ai_xmp.png")
    dst = parse_xmp_digital_source_type(img_path)
    assert dst == "trainedAlgorithmicMedia"

    res = analyze_provenance(img_path)
    assert res["name"] == "Provenance"
    assert res["available"] is True
    assert res["score"] >= 0.90
    assert any("AI-generated" in f or "trainedAlgorithmicMedia" in f for f in res["flags"])


def test_xmp_camera_declared(tmp_path):
    """Real fixture: XMP DigitalSourceType declaring digitalCapture (camera)."""
    img_path = create_xmp_png(tmp_path, "digitalCapture", "camera_xmp.png")
    dst = parse_xmp_digital_source_type(img_path)
    assert dst == "digitalCapture"

    res = analyze_provenance(img_path)
    assert res["name"] == "Provenance"
    assert res["available"] is True
    assert res["score"] == 0.0
    assert any("authentic capture" in f or "camera" in f for f in res["flags"])
    assert res["details"]["is_camera_verified"] is True


def test_no_credentials_neutral(tmp_path):
    """Real fixture: Plain image without credentials or XMP."""
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (50, 50), color=(200, 200, 200)).save(str(p))

    res = analyze_provenance(str(p))
    assert res["name"] == "Provenance"
    assert res["available"] is True
    assert res["score"] == 0.0
    assert res["weight_factor"] == 0.0  # Must not dilute or shift fusion
    assert any("typical for standard photos" in f for f in res["flags"])


def test_c2pa_ai_declared_mock(tmp_path):
    """Mock test: C2PA manifest declaring generative AI action and generator."""
    p = tmp_path / "c2pa_ai.jpg"
    Image.new("RGB", (50, 50)).save(str(p))

    mock_manifest = {
        "active_manifest": "urn:c2pa:manifest_1",
        "manifests": {
            "urn:c2pa:manifest_1": {
                "claim_generator": "Adobe Firefly 2.0",
                "signature_info": {"issuer": "Adobe Inc.", "time": "2026-10-20"},
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "softwareAgent": "Adobe Firefly",
                                    "digitalSourceType": "trainedAlgorithmicMedia"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }

    with patch("verity.provenance._extract_c2pa_info", return_value=(True, mock_manifest, True)):
        res = analyze_provenance(str(p))
        assert res["available"] is True
        assert res["score"] >= 0.90
        assert any("AI-generated" in f for f in res["flags"])
        assert res["details"]["claim_generator"] == "Adobe Firefly 2.0"
        assert res["details"]["signer"] == "Adobe Inc."

        # Verify decisive fusion behavior
        verdict = fuse([res])
        assert verdict["band"] == BAND_STRONG
        assert any("Provenance" in r for r in verdict["reasons"])


def test_c2pa_camera_declared_mock(tmp_path):
    """Mock test: Valid C2PA manifest from a verified hardware camera device."""
    p = tmp_path / "c2pa_camera.jpg"
    Image.new("RGB", (50, 50)).save(str(p))

    mock_manifest = {
        "active_manifest": "urn:c2pa:camera_1",
        "manifests": {
            "urn:c2pa:camera_1": {
                "claim_generator": "Leica M11 Camera Hardware",
                "signature_info": {"issuer": "Leica Camera AG CA", "time": "2026-10-24"},
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "softwareAgent": "Leica M11 Firmware",
                                    "digitalSourceType": "digitalCapture"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }

    with patch("verity.provenance._extract_c2pa_info", return_value=(True, mock_manifest, True)):
        res = analyze_provenance(str(p))
        assert res["available"] is True
        assert res["score"] == 0.0
        assert res["details"]["is_camera_verified"] is True
        assert any("authentic capture" in f for f in res["flags"])

        # Fusion: clean checks + camera provenance -> High confidence
        clean_checks = [
            {"name": "Metadata", "score": 0.0, "available": True, "flags": []},
            {"name": "Copy-Move", "score": 0.0, "available": True, "flags": []},
            res
        ]
        verdict = fuse(clean_checks)
        assert verdict["band"] == BAND_CLEAN
        assert verdict["confidence"] == "High"
        assert any("Provenance verified" in r for r in verdict["reasons"])

        # Fusion: visual check flagged (e.g. Copy-Move 0.85) -> Must NEVER turn Clean
        flagged_checks = [
            {"name": "Metadata", "score": 0.0, "available": True, "flags": []},
            {"name": "Copy-Move", "score": 0.88, "available": True, "flags": ["Duplication detected"]},
            res
        ]
        verdict_flagged = fuse(flagged_checks)
        assert verdict_flagged["band"] == BAND_STRONG
        assert any("conflicting manipulation indicators" in r for r in verdict_flagged["reasons"])


def test_c2pa_invalid_signature_mock(tmp_path):
    """Mock test: C2PA manifest with invalid signature / post-signing tampering."""
    p = tmp_path / "c2pa_tampered.jpg"
    Image.new("RGB", (50, 50)).save(str(p))

    mock_manifest = {
        "active_manifest": "urn:c2pa:tampered",
        "manifests": {
            "urn:c2pa:tampered": {
                "claim_generator": "Photoshop",
                "signature_info": {"issuer": "Unknown", "time": "2026-10-20"},
                "validation_status": [{"code": "assertion.dataHash.mismatch", "explanation": "Hash mismatch"}]
            }
        }
    }

    with patch("verity.provenance._extract_c2pa_info", return_value=(True, mock_manifest, False)):
        res = analyze_provenance(str(p))
        assert res["available"] is True
        assert 0.65 <= res["score"] <= 0.75
        assert any("INVALID" in f or "modified after signing" in f for f in res["flags"])
        assert res["details"]["signature_valid"] is False


def test_c2pa_library_missing_fallback(tmp_path):
    """When c2pa-python is not installed, gracefully fall back with available=False."""
    p = tmp_path / "no_lib.jpg"
    Image.new("RGB", (50, 50)).save(str(p))

    with patch("verity.provenance.C2PA_AVAILABLE", False), patch("verity.provenance.c2pa", None):
        res = analyze_provenance(str(p))
        assert res["available"] is False
        assert res["score"] == 0.0
        assert any("not installed" in f for f in res["flags"])


def test_findings_escaping_and_report_rendering(tmp_path):
    """Test that malicious provenance text is HTML-escaped in findings and rendered in PDF."""
    malicious_res = {
        "name": "Provenance",
        "score": 0.95,
        "available": True,
        "flags": ["AI origin <script>alert('pwn')</script>"],
        "details": {
            "claim_generator": "<img src=x onerror=alert(1)>",
            "signer": "<b>FakeCA</b>",
            "actions": ["c2pa.created <script>"],
            "signature_valid": True,
        }
    }

    # Test HTML escaping in findings_md
    md_output = findings_md([malicious_res])
    assert "<script>" not in md_output
    assert "&lt;script&gt;" in md_output
    assert "&lt;img" in md_output

    # Test PDF report generation
    verdict = {"band": BAND_STRONG, "score": 0.95, "confidence": "High", "reasons": ["AI origin declared"]}
    pdf_path = build_report(
        file_name="suspicious.jpg",
        kind="Image",
        verdict=verdict,
        results=[malicious_res],
        metadata={},
        mask_gps=True,
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000
