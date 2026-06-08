"""
Integration Tests for Docling Custom OCR Service
=================================================

Covers two test modes:
  A) HTTP API tests  — hit the running FastAPI server (requires server up)
  B) Plugin tests    — exercise the OCR plugin directly (no server needed)

SETUP
-----
    conda activate raas2
    cd c:\\wajahat\\personal\\learning\\docling-custom-ocr

    # Install the OCR plugin in editable mode (once)
    pip install -e src/custom-ocr

    # Optional: put sample files for file-upload tests
    mkdir tests\\samples
    copy a_scanned.pdf tests\\samples\\
    copy a_text.pdf    tests\\samples\\
    copy an_image.png  tests\\samples\\
    copy a_doc.docx    tests\\samples\\

RUN (server must be started separately first)
---------------------------------------------
    # Terminal 1 — start server
    python src/main.py

    # Terminal 2 — run all tests
    pytest tests/test_integration.py -v

    # Run only plugin tests (no server required)
    pytest tests/test_integration.py -v -m plugin

    # Run only server-dependent API tests
    pytest tests/test_integration.py -v -m api

    # Quick smoke-run as a plain script
    python tests/test_integration.py
"""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("API_URL", "http://localhost:8100")
SAMPLES_DIR = Path(__file__).parent / "samples"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _server_is_up() -> bool:
    """Return True if the server responds to /health."""
    try:
        import requests
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _read_sample(filename: str) -> bytes:
    path = SAMPLES_DIR / filename
    if not path.exists():
        pytest.skip(f"Sample file not found: {path}")
    return path.read_bytes()


def _upload_file(filename: str, content: bytes, **form_fields):
    """POST a file to /process-document/ and return the response."""
    import requests
    return requests.post(
        f"{BASE_URL}/process-document/",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
        data=form_fields,
    )


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
#   @pytest.mark.api    — requires running server
#   @pytest.mark.plugin — exercises the plugin classes directly, no server

pytestmark = []  # file-level markers applied per class/function below


# ===========================================================================
# A. HTTP API TESTS  (pytest -m api)
# ===========================================================================

@pytest.mark.api
class TestHealthEndpoint:
    """Server liveness probe."""

    def setup_method(self):
        if not _server_is_up():
            pytest.skip("Server not running at " + BASE_URL)

    def test_health_returns_200(self):
        import requests
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self):
        import requests
        r = requests.get(f"{BASE_URL}/health")
        assert r.json() == {"status": "ok"}


@pytest.mark.api
class TestRawTextAPI:
    """Send plain text; server chunks it without Docling."""

    def setup_method(self):
        if not _server_is_up():
            pytest.skip("Server not running at " + BASE_URL)

    def test_hierarchical_chunking_returns_200(self):
        import requests
        r = requests.post(
            f"{BASE_URL}/process-document/",
            data={
                "text": "First sentence. Second sentence. Third one here.",
                "chunking_strategy": "hierarchical",
            },
        )
        assert r.status_code == 200

    def test_hierarchical_response_structure(self):
        import requests
        r = requests.post(
            f"{BASE_URL}/process-document/",
            data={"text": "Hello world. This is a test."},
        )
        body = r.json()
        assert body["status"] == "success"
        assert body["source"] == "raw_text"
        assert isinstance(body["chunks"], list)
        assert len(body["chunks"]) > 0
        # Every chunk must carry these keys
        for chunk in body["chunks"]:
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert "meta" in chunk

    def test_recursive_chunking_returns_200(self):
        import requests
        r = requests.post(
            f"{BASE_URL}/process-document/",
            data={
                "text": "Para one.\n\nPara two.\n\nPara three.",
                "chunking_strategy": "recursive",
            },
        )
        assert r.status_code == 200
        assert r.json()["chunking_strategy"] == "recursive"

    def test_long_text_produces_multiple_chunks(self):
        import requests
        long_text = ("This is sentence number one. " * 50).strip()
        r = requests.post(
            f"{BASE_URL}/process-document/",
            data={"text": long_text, "chunking_strategy": "hierarchical"},
        )
        body = r.json()
        assert body["total_chunks"] > 1

    def test_unsupported_strategy_returns_400(self):
        import requests
        r = requests.post(
            f"{BASE_URL}/process-document/",
            data={"text": "Hello.", "chunking_strategy": "semantic"},
        )
        assert r.status_code == 400
        assert "unsupported" in r.json()["detail"].lower()

    def test_whitespace_only_text_returns_400(self):
        import requests
        r = requests.post(
            f"{BASE_URL}/process-document/",
            data={"text": "   "},
        )
        assert r.status_code == 400

    def test_both_file_and_text_returns_400(self):
        import requests
        r = requests.post(
            f"{BASE_URL}/process-document/",
            data={"text": "some text"},
            files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert r.status_code == 400
        assert "not both" in r.json()["detail"].lower()

    def test_no_input_returns_400(self):
        import requests
        r = requests.post(f"{BASE_URL}/process-document/")
        assert r.status_code == 400

    def test_unsupported_extension_returns_400(self):
        import requests
        r = requests.post(
            f"{BASE_URL}/process-document/",
            files={"file": ("malware.exe", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert r.status_code == 400
        assert ".pdf" in r.json()["detail"]


@pytest.mark.api
class TestPredictEndpoint:
    """/predict — the JSON contract used by the AI Gateway."""

    def setup_method(self):
        if not _server_is_up():
            pytest.skip("Server not running at " + BASE_URL)

    def test_predict_with_text(self):
        import requests
        payload = {
            "text": "Hello from the predict endpoint.",
            "strategy": "hierarchical",
            "chunk_size": 500,
            "chunk_overlap": 50,
        }
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "chunks" in body
        assert len(body["chunks"]) > 0
        chunk = body["chunks"][0]
        assert "chunk_index" in chunk
        assert "content" in chunk
        assert "metadata" in chunk

    def test_predict_recursive_strategy(self):
        import requests
        payload = {
            "text": "Para one.\n\nPara two.\n\nPara three.",
            "strategy": "recursive",
            "chunk_size": 200,
            "chunk_overlap": 20,
        }
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 200

    def test_predict_all_strategy_enum_values(self):
        """All 5 ChunkingStrategy values must be accepted (mapped to 2 backends)."""
        import requests
        strategies = ["fixed", "recursive", "hierarchical", "document_structure", "semantic"]
        for strategy in strategies:
            payload = {
                "text": "Test sentence.",
                "strategy": strategy,
                "chunk_size": 500,
                "chunk_overlap": 0,
            }
            r = requests.post(f"{BASE_URL}/predict", json=payload)
            assert r.status_code == 200, f"strategy={strategy!r} returned {r.status_code}"

    def test_predict_no_input_returns_400(self):
        import requests
        payload = {"strategy": "hierarchical", "chunk_size": 500, "chunk_overlap": 0}
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 400


@pytest.mark.api
class TestFileUploadAPI:
    """Upload actual files — skipped if sample files are not present."""

    def setup_method(self):
        if not _server_is_up():
            pytest.skip("Server not running at " + BASE_URL)

    def test_pdf_no_ocr(self):
        content = _read_sample("sample_text.pdf")
        r = _upload_file(
            "sample_text.pdf", content,
            use_ocr="false",
            chunking_strategy="hierarchical",
        )
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "file"
        assert body["use_ocr"] is False
        assert len(body["chunks"]) > 0

    def test_pdf_with_ocr(self):
        """Uses whatever OCR_BACKEND the server has configured."""
        content = _read_sample("sample_scan.pdf")
        r = _upload_file(
            "sample_scan.pdf", content,
            use_ocr="true",
            chunking_strategy="hierarchical",
        )
        assert r.status_code == 200
        assert len(r.json()["chunks"]) >= 0  # even empty result is valid

    def test_image_always_uses_ocr(self):
        content = _read_sample("sample.png")
        r = _upload_file("sample.png", content, use_ocr="false")
        assert r.status_code == 200

    def test_docx_standard_extraction(self):
        content = _read_sample("sample.docx")
        r = _upload_file("sample.docx", content)
        assert r.status_code == 200
        assert r.json()["source"] == "file"

    def test_txt_standard_extraction(self):
        content = _read_sample("sample.txt")
        r = _upload_file("sample.txt", content)
        assert r.status_code == 200

    def test_filename_echoed_in_response(self):
        content = _read_sample("sample_text.pdf")
        r = _upload_file("sample_text.pdf", content, use_ocr="false")
        assert r.json()["filename"] == "sample_text.pdf"

    def test_response_includes_total_chunks(self):
        content = _read_sample("sample_text.pdf")
        r = _upload_file("sample_text.pdf", content, use_ocr="false")
        body = r.json()
        assert "total_chunks" in body
        assert body["total_chunks"] == len(body["chunks"])


# ===========================================================================
# B. PLUGIN TESTS  (pytest -m plugin) — no server required
# ===========================================================================

@pytest.mark.plugin
class TestCustomApiOcrOptions:
    """Validate that CustomApiOcrOptions constructs correctly."""

    def test_kind_is_custom_api(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrOptions
        opts = CustomApiOcrOptions(url="http://localhost:9999/ocr")
        assert opts.kind == "custom_api"

    def test_default_lang(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrOptions
        opts = CustomApiOcrOptions(url="http://localhost:9999/ocr")
        assert opts.lang == ["en"]

    def test_default_timeout(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrOptions
        opts = CustomApiOcrOptions(url="http://localhost:9999/ocr")
        assert opts.timeout == 300.0

    def test_default_scale(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrOptions
        opts = CustomApiOcrOptions(url="http://localhost:9999/ocr")
        assert opts.scale == 3.0

    def test_custom_lang(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrOptions
        opts = CustomApiOcrOptions(url="http://localhost:9999/ocr", lang=["ar", "en"])
        assert opts.lang == ["ar", "en"]

    def test_missing_url_raises(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrOptions
        with pytest.raises(Exception):
            CustomApiOcrOptions()

    def test_get_options_type(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel, CustomApiOcrOptions
        assert CustomApiOcrModel.get_options_type() is CustomApiOcrOptions


@pytest.mark.plugin
class TestCustomApiOcrModelHelpers:
    """Test internal helper methods without touching the network."""

    def _make_model(self, url="http://localhost:9999/ocr", enabled=False):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel, CustomApiOcrOptions
        opts = CustomApiOcrOptions(url=url, lang=["en"])
        return CustomApiOcrModel(
            enabled=enabled,
            artifacts_path=None,
            options=opts,
            accelerator_options=MagicMock(),
        )

    # --- _get_confidence_value -------------------------------------------

    def test_get_confidence_float(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        assert CustomApiOcrModel._get_confidence_value(0.95) == pytest.approx(0.95)

    def test_get_confidence_string(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        assert CustomApiOcrModel._get_confidence_value("0.8") == pytest.approx(0.8)

    def test_get_confidence_none_returns_zero(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        assert CustomApiOcrModel._get_confidence_value(None) == 0.0

    def test_get_confidence_invalid_returns_zero(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        assert CustomApiOcrModel._get_confidence_value("bad") == 0.0

    # --- _map_bbox_to_page -----------------------------------------------

    def test_map_bbox_valid(self):
        from docling_core.types.doc import BoundingBox, CoordOrigin
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model()
        ocr_rect = BoundingBox(l=10, t=20, r=100, b=200, coord_origin=CoordOrigin.TOPLEFT)
        result = model._map_bbox_to_page([0, 0, 90, 90], ocr_rect)
        assert result is not None
        assert result.l == pytest.approx(10.0)
        assert result.t == pytest.approx(20.0)

    def test_map_bbox_zero_area_returns_none(self):
        from docling_core.types.doc import BoundingBox, CoordOrigin
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model()
        ocr_rect = BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.TOPLEFT)
        assert model._map_bbox_to_page([50, 50, 50, 50], ocr_rect) is None

    # --- _iter_prediction_candidates -------------------------------------

    def test_simple_prediction_parsed(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model()
        pred = {"text": "Hello", "bbox": [0, 0, 100, 20], "confidence": 0.99}
        results = list(model._iter_prediction_candidates(pred))
        assert len(results) == 1
        text, bbox, conf = results[0]
        assert text == "Hello"
        assert conf == pytest.approx(0.99)

    def test_empty_text_skipped(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model()
        pred = {"text": "  ", "bbox": [0, 0, 100, 20], "confidence": 0.9}
        assert list(model._iter_prediction_candidates(pred)) == []

    def test_missing_bbox_returns_empty(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model()
        pred = {"text": "Hello"}
        assert list(model._iter_prediction_candidates(pred)) == []

    def test_nested_json_in_text_field(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model()
        nested = json.dumps([{"text": "Word", "bbox": [0, 0, 50, 10], "confidence": 0.8}])
        pred = {"text": nested, "confidence": 0.5}
        results = list(model._iter_prediction_candidates(pred))
        assert len(results) == 1
        assert results[0][0] == "Word"

    # --- _extract_cells_from_document_payload ----------------------------

    def test_cells_data_top_level(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        payload = {
            "cells_data": [
                {"text": "A", "bbox": [0, 0, 10, 10]},
                {"text": "B", "bbox": [10, 0, 20, 10]},
            ]
        }
        cells = CustomApiOcrModel._extract_cells_from_document_payload(payload)
        assert cells is not None
        assert len(cells) == 2

    def test_page_results_structure(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        payload = {
            "page_results": [
                {"cells_data": [{"text": "P1", "bbox": [0, 0, 10, 10]}]},
                {"cells_data": [{"text": "P2", "bbox": [20, 0, 30, 10]}]},
            ]
        }
        cells = CustomApiOcrModel._extract_cells_from_document_payload(payload)
        assert cells is not None
        assert len(cells) == 2

    def test_unknown_format_returns_none(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        payload = {"text": "hello", "confidence": 0.9}
        result = CustomApiOcrModel._extract_cells_from_document_payload(payload)
        assert result is None

    # --- _call_api (mocked network) --------------------------------------

    def test_call_api_returns_list_on_success(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model(enabled=True)
        fake_image = MagicMock()
        fake_image.mode = "RGB"

        fake_response = MagicMock()
        fake_response.json.return_value = [
            {"text": "Hello", "bbox": [0, 0, 100, 20], "confidence": 0.9}
        ]

        with patch.object(model._session, "post", return_value=fake_response):
            result = model._call_api(fake_image)

        assert isinstance(result, list)
        assert result[0]["text"] == "Hello"

    def test_call_api_returns_empty_on_http_error(self):
        import requests as req_lib
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model(enabled=True)
        fake_image = MagicMock()
        fake_image.mode = "RGB"

        with patch.object(model._session, "post", side_effect=req_lib.ConnectionError("down")):
            result = model._call_api(fake_image)

        assert result == []

    def test_call_api_returns_empty_on_invalid_json(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        model = self._make_model(enabled=True)
        fake_image = MagicMock()
        fake_image.mode = "RGB"

        fake_response = MagicMock()
        fake_response.json.side_effect = ValueError("bad json")
        fake_response.text = "not json"

        with patch.object(model._session, "post", return_value=fake_response):
            result = model._call_api(fake_image)

        assert result == []


@pytest.mark.plugin
class TestPluginRegistration:
    """Verify the Docling plugin hook is wired correctly."""

    def test_ocr_engines_hook_returns_dict(self):
        from docling_custom_ocr.plugin import ocr_engines
        result = ocr_engines()
        assert isinstance(result, dict)
        assert "ocr_engines" in result

    def test_ocr_engines_hook_contains_model_class(self):
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        from docling_custom_ocr.plugin import ocr_engines
        engines = ocr_engines()["ocr_engines"]
        assert CustomApiOcrModel in engines


# ===========================================================================
# C. DOCLING PIPELINE TEST  (integration, needs GPU or CPU inference)
# ===========================================================================

@pytest.mark.slow
class TestDoclingPipelineWithCustomOcr:
    """
    Full pipeline test — converts a real PDF using the custom OCR plugin.
    Requires a running OCR API endpoint (set CUSTOM_OCR_API_URL env var).

    Skip with: pytest -m "not slow"
    """

    def test_pipeline_with_mocked_ocr_api(self, tmp_path):
        """Run a real Docling conversion but mock the HTTP OCR call."""
        from docling_custom_ocr.api_ocr import CustomApiOcrModel, CustomApiOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        sample_pdf = SAMPLES_DIR / "sample_scan.pdf"
        if not sample_pdf.exists():
            pytest.skip(f"No sample PDF at {sample_pdf}")

        opts = CustomApiOcrOptions(url="http://mock-ocr.test/ocr", lang=["en"])
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            allow_external_plugins=True,
            ocr_options=opts,
            force_full_page_ocr=True,
        )

        mock_api_response = [{"text": "Mock OCR text", "bbox": [0, 0, 100, 20], "confidence": 0.95}]

        with patch.object(CustomApiOcrModel, "_call_api", return_value=mock_api_response):
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            result = converter.convert(str(sample_pdf))

        md = result.document.export_to_markdown()
        assert isinstance(md, str)


# ===========================================================================
# SCRIPT MODE — run with:  python tests/test_integration.py
# ===========================================================================

def _run_quick_smoke():
    """Quick non-pytest smoke checks for script mode."""
    print("\n" + "=" * 60)
    print("DOCLING CUSTOM OCR — SMOKE TEST")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(label, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  [PASS] {label}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {label}")
            print(f"         {exc}")
            failed += 1

    # --- Plugin-level checks (always run, no server needed) ---------------
    print("\n--- Plugin checks (no server needed) ---")

    def chk_options():
        from docling_custom_ocr.api_ocr import CustomApiOcrOptions
        o = CustomApiOcrOptions(url="http://localhost:9/ocr")
        assert o.kind == "custom_api"
        assert o.timeout == 300.0

    def chk_confidence():
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        assert CustomApiOcrModel._get_confidence_value(0.9) == 0.9
        assert CustomApiOcrModel._get_confidence_value(None) == 0.0

    def chk_plugin():
        from docling_custom_ocr.plugin import ocr_engines
        from docling_custom_ocr.api_ocr import CustomApiOcrModel
        assert CustomApiOcrModel in ocr_engines()["ocr_engines"]

    def chk_parse_simple_prediction():
        from docling_custom_ocr.api_ocr import CustomApiOcrModel, CustomApiOcrOptions
        m = CustomApiOcrModel(
            enabled=False, artifacts_path=None,
            options=CustomApiOcrOptions(url="http://x/ocr"),
            accelerator_options=MagicMock(),
        )
        res = list(m._iter_prediction_candidates(
            {"text": "Hi", "bbox": [0, 0, 10, 10], "confidence": 0.9}
        ))
        assert res[0][0] == "Hi"

    check("CustomApiOcrOptions constructs", chk_options)
    check("_get_confidence_value", chk_confidence)
    check("Plugin hook registration", chk_plugin)
    check("Prediction parsing", chk_parse_simple_prediction)

    # --- API checks (only if server is up) --------------------------------
    print("\n--- API checks (server at " + BASE_URL + ") ---")
    if not _server_is_up():
        print("  [SKIP] Server not running — skipping API checks")
        print("         Start with: python src/main.py")
    else:
        import requests

        def chk_health():
            r = requests.get(f"{BASE_URL}/health")
            assert r.status_code == 200

        def chk_raw_text():
            r = requests.post(
                f"{BASE_URL}/process-document/",
                data={"text": "Hello world. This is a test sentence."},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "success"
            assert len(body["chunks"]) > 0

        def chk_recursive():
            r = requests.post(
                f"{BASE_URL}/process-document/",
                data={
                    "text": "Para one.\n\nPara two.",
                    "chunking_strategy": "recursive",
                },
            )
            assert r.status_code == 200

        def chk_predict():
            r = requests.post(
                f"{BASE_URL}/predict",
                json={
                    "text": "Hello from predict.",
                    "strategy": "hierarchical",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                },
            )
            assert r.status_code == 200
            assert "chunks" in r.json()

        def chk_bad_ext():
            r = requests.post(
                f"{BASE_URL}/process-document/",
                files={"file": ("bad.exe", io.BytesIO(b"x"), "application/octet-stream")},
            )
            assert r.status_code == 400

        check("GET /health", chk_health)
        check("POST /process-document/ (raw text)", chk_raw_text)
        check("POST /process-document/ (recursive)", chk_recursive)
        check("POST /predict (text)", chk_predict)
        check("POST /process-document/ (bad extension → 400)", chk_bad_ext)

    # --- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed", "✓" if failed == 0 else "✗")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    # If pytest is available, delegate to it for proper reporting.
    # Otherwise fall back to the built-in smoke runner.
    try:
        import pytest as _pytest
        sys.exit(_pytest.main([__file__, "-v", "--tb=short"]))
    except SystemExit:
        raise
    except Exception:
        failures = _run_quick_smoke()
        sys.exit(0 if failures == 0 else 1)
