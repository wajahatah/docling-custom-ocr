"""
Quick file upload tester.

Usage:
    conda activate raas2
    python tests/test_upload.py                          # runs all built-in demos
    python tests/test_upload.py path/to/file.pdf         # upload a specific file
    python tests/test_upload.py path/to/file.pdf surya   # use surya backend for this run

Chunks are saved automatically by the server to chunks/{filename}_chunks.json.
"""

import os
import sys

import requests

BASE_URL = os.getenv("API_URL", "http://localhost:8100")

# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _print_result(label: str, response: requests.Response):
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"  Status : {response.status_code}")
    try:
        body = response.json()
        chunks = body.get("chunks", [])
        print(f"  Chunks : {body.get('total_chunks', len(chunks))}")
        print(f"  Source : {body.get('source', body.get('status', '?'))}")
        if chunks:
            preview = chunks[0].get("text") or chunks[0].get("content", "")
            print(f"  First  : {preview[:120]!r}")
    except Exception:
        print(f"  Body   : {response.text[:200]}")
    print(f"{'─' * 55}")


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────
# individual test functions
# ─────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print(f"\n[Health]  {r.status_code}  {r.json()}")


def test_raw_text():
    r = requests.post(
        f"{BASE_URL}/process-document/",
        data={
            "text": (
                "Docling is a document-understanding pipeline. "
                "It parses PDFs, images, and office documents. "
                "OCR is performed on scanned pages. "
                "The output can be exported as Markdown or JSON."
            ),
            "chunking_strategy": "hierarchical",
        },
    )
    _print_result("Raw text  →  hierarchical", r)


def test_raw_text_recursive():
    r = requests.post(
        f"{BASE_URL}/process-document/",
        data={
            "text": "Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            "chunking_strategy": "recursive",
        },
    )
    _print_result("Raw text  →  recursive", r)


def test_upload_file(file_path: str, use_ocr: bool = True, strategy: str = "hierarchical"):
    """Upload any file from disk. The server saves chunks to chunks/{stem}_chunks.json."""
    if not os.path.exists(file_path):
        print(f"\n[SKIP] File not found: {file_path}")
        return

    ext  = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)
    size = os.path.getsize(file_path)

    print(f"\n[Upload] {name}  ({size:,} bytes)  use_ocr={use_ocr}  strategy={strategy}")

    with open(file_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/process-document/",
            files={"file": (name, f, "application/octet-stream")},
            data={"use_ocr": str(use_ocr).lower(), "chunking_strategy": strategy},
            timeout=300,
        )
    _print_result(f"File upload  ({ext})", r)


def test_predict_text():
    """Test the /predict gateway endpoint."""
    r = requests.post(
        f"{BASE_URL}/predict",
        json={
            "text": "The quick brown fox jumps over the lazy dog.",
            "strategy": "recursive",
            "chunk_size": 100,
            "chunk_overlap": 10,
        },
        timeout=30,
    )
    print(f"\n{'─' * 55}")
    print("  /predict  →  text input")
    print(f"  Status : {r.status_code}")
    body = r.json()
    chunks = body.get("chunks", [])
    print(f"  Chunks : {len(chunks)}")
    if chunks:
        print(f"  First  : {chunks[0].get('content', '')!r}")
    print(f"{'─' * 55}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nTarget server : {BASE_URL}")
    print(f"Swagger UI    : {BASE_URL}/docs\n")

    if not _server_up():
        print("ERROR: Server is not running.")
        print(f"       Start it with:  python src/main.py")
        sys.exit(1)

    # ── user passed a file path on the command line ──────────────────────
    if len(sys.argv) >= 2:
        file_path  = sys.argv[1]
        ocr_backend = sys.argv[2] if len(sys.argv) >= 3 else None

        if ocr_backend:
            os.environ["OCR_BACKEND"] = ocr_backend
            print(f"[Info] Overriding OCR_BACKEND={ocr_backend} for this run")

        test_upload_file(file_path, use_ocr=True,  strategy="hierarchical")
        test_upload_file(file_path, use_ocr=False, strategy="recursive")
        sys.exit(0)

    # ── no args — run built-in demos ─────────────────────────────────────
    test_health()
    test_raw_text()
    test_raw_text_recursive()
    test_predict_text()

    # Sample files — skip gracefully if not present
    samples = [
        ("tests/samples/sample_text.pdf",  True,  "hierarchical"),
        ("tests/samples/sample_scan.pdf",  True,  "hierarchical"),
        ("tests/samples/sample.png",       True,  "hierarchical"),
        ("tests/samples/sample.docx",      False, "hierarchical"),
        ("tests/samples/sample.txt",       False, "recursive"),
    ]
    for path, ocr, strategy in samples:
        test_upload_file(path, use_ocr=ocr, strategy=strategy)

    print("\nDone. Open the Swagger UI for interactive testing:")
    print(f"  {BASE_URL}/docs\n")
