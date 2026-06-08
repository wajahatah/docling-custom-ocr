# Docling Custom OCR — Complete Integration Guide

This repository shows how to plug a **custom HTTP OCR service** into [Docling](https://github.com/DS4SD/docling) without touching Docling's source code, and wraps it in a production-ready **FastAPI chunking service** that the AI Gateway calls.

---

## Table of Contents

1. [What This Does](#1-what-this-does)
2. [Project Structure](#2-project-structure)
3. [Quick Start](#3-quick-start)
4. [How Docling's Plugin System Works](#4-how-doclings-plugin-system-works)
5. [The OCR Plugin — Step by Step](#5-the-ocr-plugin--step-by-step)
   - 5.1 [CustomApiOcrOptions](#51-customapiocroptions)
   - 5.2 [CustomApiOcrModel](#52-customapiocrmodel)
   - 5.3 [Plugin Registration](#53-plugin-registration)
6. [The FastAPI Chunking Service](#6-the-fastapi-chunking-service)
   - 6.1 [File Routing](#61-file-routing)
   - 6.2 [Chunking Strategies](#62-chunking-strategies)
   - 6.3 [API Endpoints](#63-api-endpoints)
7. [Integration Guide](#7-integration-guide)
   - 7.1 [Direct Python Usage](#71-direct-python-usage)
   - 7.2 [Via the FastAPI Service (HTTP)](#72-via-the-fastapi-service-http)
   - 7.3 [Via Docker](#73-via-docker)
   - 7.4 [Embed in an Existing Project](#74-embed-in-an-existing-project)
8. [OCR API Response Formats](#8-ocr-api-response-formats)
9. [Environment Variables](#9-environment-variables)
   - 9.1 [The `.env` file](#91-the-env-file)
   - 9.2 [All variables](#92-all-variables)
   - 9.3 [Switching between backends](#93-switching-between-backends)
   - 9.4 [How values flow into the code](#94-how-values-flow-into-the-code)
   - 9.5 [Precedence](#95-precedence)
   - 9.6 [Docker](#96-docker)
10. [Testing](#10-testing)
    - 10.1 [Swagger UI — interactive file upload](#101-swagger-ui--interactive-file-upload-browser)
    - 10.2 [`test_upload.py` — terminal script](#102-test_uploadpy--terminal-script)
    - 10.3 [Unit tests](#103-unit-tests--fast-no-server-all-deps-mocked)
    - 10.4 [Plugin tests](#104-plugin-tests--no-server-needed)
    - 10.5 [API tests](#105-api-tests--requires-running-server)
    - 10.6 [Full pipeline test](#106-full-pipeline-test--real-docling-inference-slow)
    - 10.7 [Quick smoke test](#107-quick-smoke-test--plain-script-no-pytest)
11. [Common Pitfalls](#11-common-pitfalls)
12. [How It Compares to the Built-in OCR Engines](#12-how-it-compares-to-the-built-in-ocr-engines)

---

## 1. What This Does

Docling is a document-understanding pipeline: PDF parsing → OCR → layout detection → table/figure detection → export (Markdown, JSON, etc.).

By default, Docling ships with [Surya OCR](https://github.com/VikParuchuri/surya) and Tesseract. This project replaces the OCR step with **your own HTTP endpoint** — useful when you have a proprietary OCR service, a GPU cluster running a custom model, or an Arabic/mixed-language API.

The approach:

* A separate Python package (`docling-custom-ocr`) registers as a **Docling external plugin**.
* Docling's factory discovers it and calls it exactly like a built-in engine.
* A **FastAPI service** (`src/main.py`) wraps the whole pipeline and exposes it as HTTP endpoints for the AI Gateway.

No Docling source code is modified.

---

## 2. Project Structure

```
docling-custom-ocr/
│
├── README.md                    ← this file
├── requirements.txt             ← all Python deps (docling, torch, surya, fastapi…)
├── Dockerfile                   ← production container (python:3.12-slim, port 8100)
├── pytest.ini                   ← test markers: api, plugin, slow
├── .env                         ← local config (gitignored — never committed)
├── .env.example                 ← template to copy and fill in
├── .gitignore                   ← excludes .env from git
├── test_upload.py               ← quick file-upload test script (run from terminal)
│
├── src/
│   ├── main.py                  ← FastAPI chunking service (entry point, loads .env)
│   │
│   ├── contracts/
│   │   ├── contracts.py         ← ChunkerPredictRequest / ChunkerPredictResponse
│   │   └── schemas.py           ← ChunkingStrategy enum (5 values → 2 backends)
│   │
│   └── custom-ocr/              ← the installable OCR plugin package
│       ├── pyproject.toml       ← registers the docling entry-point
│       └── src/
│           └── docling_custom_ocr/
│               ├── __init__.py
│               ├── plugin.py    ← hook Docling calls at discovery
│               └── api_ocr.py   ← CustomApiOcrOptions + CustomApiOcrModel
│
└── tests/
    ├── test_main.py             ← unit tests (all dependencies mocked, fast)
    └── test_integration.py      ← integration tests (plugin + live server)
```

---

## 3. Quick Start

### Prerequisites

* [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
* A running OCR HTTP service (or use `OCR_BACKEND=surya` to skip the custom API)

### Step 1 — activate the environment

```powershell
conda activate raas2
```

### Step 2 — configure your `.env`

Copy the example file and fill in your values:

```powershell
cd c:\wajahat\personal\learning\docling-custom-ocr
copy .env.example .env
```

Then open `.env` and set your OCR endpoint and preferred backend:

```ini
# Choose the OCR engine: custom_api  or  surya
OCR_BACKEND=custom_api

# Your OCR service URL (only used when OCR_BACKEND=custom_api)
CUSTOM_OCR_API_URL=https://your-ocr-service.example.com/ocr

# Request timeout in seconds
CUSTOM_OCR_TIMEOUT=300

# Port the FastAPI service listens on
SERVICE_PORT=8100
```

To use Surya instead (no external OCR API needed), just set:

```ini
OCR_BACKEND=surya
```

### Step 3 — install the plugin

The plugin must be installed as a Python package so Docling can discover it via its entry-point system.

```powershell
pip install -e src/custom-ocr
```

### Step 4 — run the service

```powershell
python src/main.py
```

`main.py` automatically loads `.env` at startup via `python-dotenv`. No manual `$env:` exports needed.

Service starts at `http://localhost:8100` (or whatever `SERVICE_PORT` is set to).

### Step 5 — test it

Three options — pick whichever fits you best.

#### Option A — Swagger UI (browser, no extra setup)

Open **http://localhost:8100/docs** in your browser. FastAPI generates a full interactive UI automatically. Click any endpoint → **Try it out** → fill in the form → **Execute**. The `/process-document/` endpoint has a file picker so you can upload PDFs, images, or Office docs directly.

#### Option B — `test_upload.py` (terminal script)

```powershell
# Run all built-in demos (raw text + any files in tests/samples/)
python test_upload.py

# Upload a specific file
python test_upload.py path\to\document.pdf

# Upload with a backend override for this one run
python test_upload.py path\to\scan.pdf surya
```

The script prints status, chunk count, and a preview of the first chunk for each request. It skips sample files gracefully if they are not present.

#### Option C — curl

```powershell
# Health check
curl http://localhost:8100/health

# Raw text
curl -X POST http://localhost:8100/process-document/ `
     -F "text=Hello world." `
     -F "chunking_strategy=hierarchical"

# Upload a PDF (OCR enabled)
curl -X POST http://localhost:8100/process-document/ `
     -F "file=@scanned_invoice.pdf" `
     -F "use_ocr=true"

# Upload a PDF (text layer only, no OCR)
curl -X POST http://localhost:8100/process-document/ `
     -F "file=@my_document.pdf" `
     -F "use_ocr=false"
```

---

## 4. How Docling's Plugin System Works

### 4.1 The OCR Factory

Docling uses an internal **OCR factory** that:

1. Enumerates available OCR engines (built-in + external plugins)
2. Matches `PdfPipelineOptions.ocr_options.kind` against registered engine names
3. Instantiates the matching class with:
   ```python
   OCRModel(
       enabled=True,
       artifacts_path=Path(...),
       options=<your OcrOptions subclass>,
       accelerator_options=<AcceleratorOptions>,
   )
   ```
4. Calls the instance on each page batch:
   ```python
   for page in ocr_model(conv_res, page_batch):
       ...
   ```

### 4.2 External Plugin Discovery

Docling discovers external plugins at runtime when `allow_external_plugins=True` is set in the pipeline options. It looks for Python packages that declare the `docling` entry-point group.

The entry-point in `pyproject.toml`:

```toml
[project.entry-points."docling"]
custom_ocr_api = "docling_custom_ocr.plugin"
```

This tells Docling: *"import `docling_custom_ocr.plugin` and call its `ocr_engines()` function"*.

### 4.3 Plugin Hook

```python
# src/custom-ocr/src/docling_custom_ocr/plugin.py

from .api_ocr import CustomApiOcrModel

def ocr_engines():
    return {"ocr_engines": [CustomApiOcrModel]}
```

Docling expects this exact structure: a dict with key `"ocr_engines"` mapping to a list of **classes** (not instances).

---

## 5. The OCR Plugin — Step by Step

### 5.1 `CustomApiOcrOptions`

Every Docling OCR engine has a paired **options class** that Pydantic validates. Yours must extend `OcrOptions` and set a `kind` class-level constant that is used to look up the engine.

```python
from docling.datamodel.pipeline_options import OcrOptions
from pydantic import AnyUrl, Field, ConfigDict
from typing import ClassVar, Dict, List, Literal

class CustomApiOcrOptions(OcrOptions):
    kind: ClassVar[Literal["custom_api"]] = "custom_api"

    # Required by Docling's base class
    lang: List[str] = ["en"]

    # Your custom fields
    url: AnyUrl                            # OCR service endpoint — required
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: float = 300.0                 # seconds
    scale: float = 3.0                     # image upscale factor before sending
    confidence_threshold: float = 0.0      # discard predictions below this
    image_format: Literal["PNG", "JPEG"] = "JPEG"

    model_config = ConfigDict(extra="forbid")
```

**Critical rules:**

| Rule | Why it matters |
|------|---------------|
| `kind` must be a `ClassVar` | Docling reads it as a class attribute to match engines, not an instance field |
| `lang` must be present | The base class validates it; omitting causes a Pydantic error before your code runs |
| `extra="forbid"` | Prevents silent config typos from being ignored |

### 5.2 `CustomApiOcrModel`

Your model extends `BaseOcrModel`. The factory instantiates it with four positional keyword arguments — your constructor must accept all of them.

```python
from docling.models.base_ocr_model import BaseOcrModel
from docling.datamodel.accelerator_options import AcceleratorOptions
from pathlib import Path
from typing import Optional, Type
import requests

class CustomApiOcrModel(BaseOcrModel):

    # ------------------------------------------------------------------ #
    # Constructor — must accept exactly these kwargs from Docling's factory
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: CustomApiOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: CustomApiOcrOptions
        self.scale = self.options.scale
        self._session: Optional[requests.Session] = None

        if self.enabled:
            self._session = requests.Session()

    # ------------------------------------------------------------------ #
    # Required classmethod — tells the factory which options class to use
    # ------------------------------------------------------------------ #
    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return CustomApiOcrOptions

    # ------------------------------------------------------------------ #
    # Main entry point — called by Docling for each batch of pages
    # ------------------------------------------------------------------ #
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)   # from base class
                all_ocr_cells = []

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue

                    # Crop the page region and scale it up for better OCR
                    high_res_image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )

                    predictions = self._call_api(high_res_image)
                    cells = self._build_cells(predictions, ocr_rect)
                    all_ocr_cells.extend(cells)

                # base class method — injects cells into page
                self.post_process_cells(all_ocr_cells, page)

            yield page
```

#### How page images are extracted

Docling's `Page` object exposes its rasterized image via the backend:

```python
image = page._backend.get_page_image(scale=3.0, cropbox=ocr_rect)
# Returns a PIL Image, upscaled 3× for higher OCR accuracy
```

The `cropbox` is a `BoundingBox` from `self.get_ocr_rects(page)` — the base class identifies which page regions need OCR (image-only areas where no text layer exists).

#### How OCR cells are injected

`self.post_process_cells(cells, page)` is a **base class method**. It handles merging and registering the `TextCell` objects into the page's internal state. You do not need to mutate `page.predictions` directly — doing so bypasses Pydantic validation and crashes later pipeline stages.

**Never do this:**

```python
# WRONG — replaces Pydantic model with dict
page.predictions = {"ocr": cells}
```

**Never do this either:**

```python
# WRONG — breaks layout/table stages downstream
page.predictions = page.predictions.model_copy(update={"vlm_response": text})
```

The correct approach is `post_process_cells` — it is the only safe injection point.

#### Calling your OCR API

```python
def _call_api(self, image) -> list[dict]:
    image_format = self.options.image_format  # "JPEG" or "PNG"
    if image_format == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)

    try:
        response = self._session.post(
            str(self.options.url),
            headers=self.options.headers,
            files={"file": (f"page.{image_format.lower()}", buffer, f"image/{image_format.lower()}")},
            timeout=self.options.timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        _log.error("OCR request failed: %s", exc)
        return []

    try:
        payload = response.json()
    except ValueError:
        _log.error("OCR returned non-JSON response")
        return []

    # Handles list, flat dict, and document-style payloads
    if isinstance(payload, dict):
        cells = self._extract_cells_from_document_payload(payload)
        return cells if cells is not None else [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []
```

#### Mapping bounding boxes back to page coordinates

Your OCR API receives a **cropped, upscaled** region. Its bounding boxes are relative to that crop. The plugin maps them back to the full page coordinate space:

```python
def _map_bbox_to_page(self, bbox, ocr_rect) -> Optional[BoundingBox]:
    left, top, right, bottom = bbox

    # Undo the scale factor, then shift by the crop origin
    scaled_left   = (left   / self.scale) + ocr_rect.l
    scaled_top    = (top    / self.scale) + ocr_rect.t
    scaled_right  = (right  / self.scale) + ocr_rect.l
    scaled_bottom = (bottom / self.scale) + ocr_rect.t

    if scaled_right <= scaled_left or scaled_bottom <= scaled_top:
        return None  # degenerate box — skip

    return BoundingBox.from_tuple(
        (scaled_left, scaled_top, scaled_right, scaled_bottom),
        origin=CoordOrigin.TOPLEFT,
    )
```

### 5.3 Plugin Registration

The `pyproject.toml` inside `src/custom-ocr/` wires everything together:

```toml
[project]
name = "docling-custom-ocr"
version = "0.1.0"
dependencies = ["docling", "requests", "Pillow"]

[project.entry-points."docling"]
custom_ocr_api = "docling_custom_ocr.plugin"
```

The entry-point key (`custom_ocr_api`) is the **module path** Docling loads. Docling calls `ocr_engines()` from that module and registers whatever classes are returned.

**Installation (required before use):**

```powershell
pip install -e src/custom-ocr
```

Without this, Docling cannot find your engine and will raise a `KeyError` or silently fall back to a default.

---

## 6. The FastAPI Chunking Service

`src/main.py` wraps the OCR plugin in a FastAPI service with three extraction paths and two chunking strategies.

### 6.1 File Routing

| File type | Path taken | Notes |
|-----------|-----------|-------|
| `.png` `.jpg` `.jpeg` `.tiff` `.tif` | `extract_with_ocr()` | Always OCR; `use_ocr` flag ignored |
| `.pdf` + `use_ocr=true` (default) | `extract_with_ocr()` | Runs OCR on scanned/image PDFs |
| `.pdf` + `use_ocr=false` | `extract_pdf_without_ocr()` | Skips OCR; uses text layer only (fast) |
| `.docx` `.txt` `.pptx` `.xlsx` `.html` `.md` | `standardDocling()` | Native Docling parsers, no OCR |
| Raw text string | `chunk_raw_text()` | Bypasses Docling entirely |

### 6.2 Chunking Strategies

Two boundary algorithms, selected by the `chunking_strategy` parameter:

**`hierarchical`** — Docling's `HierarchicalChunker`
- Respects document structure (headings, sections, paragraphs)
- Best for structured documents (PDFs, DOCX with headings)
- For raw text: sentence-aware splitter with word-level fallback

**`recursive`** — LangChain `RecursiveCharacterTextSplitter`
- Tries separators in order: `\n\n` → `\n` → `. ` → ` ` → character
- Guarantees `chunk_size` is respected
- Best for plain text or when you need a fixed maximum chunk size

The `/predict` endpoint accepts a `ChunkingStrategy` enum with five values (for gateway compatibility). They map as follows:

| Enum value | Backend |
|-----------|---------|
| `fixed` | recursive |
| `recursive` | recursive |
| `hierarchical` | hierarchical |
| `document_structure` | hierarchical |
| `semantic` | hierarchical |

### 6.3 API Endpoints

#### `GET /health`

Liveness probe.

```json
{"status": "ok"}
```

#### `POST /process-document/`

Multipart form upload. For direct browser/curl use.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | — | Document to upload (mutually exclusive with `text`) |
| `text` | string | — | Raw text to chunk directly |
| `chunking_strategy` | string | `hierarchical` | `hierarchical` or `recursive` |
| `use_ocr` | bool | `true` | Enable OCR for PDFs (images always OCR) |

Response:

```json
{
  "status": "success",
  "source": "file",
  "filename": "invoice.pdf",
  "chunking_strategy": "hierarchical",
  "use_ocr": true,
  "total_chunks": 12,
  "chunks": [
    {
      "chunk_index": 0,
      "text": "Invoice #1042\nDate: 2024-01-15",
      "meta": {"source": "mock", "headings": ["Invoice"]}
    }
  ]
}
```

#### `POST /predict`

JSON contract for the AI Gateway.

Request body (`ChunkerPredictRequest`):

```json
{
  "text": "Optional inline text",
  "file_url": "https://example.com/doc.pdf",
  "read_url": "https://minio.internal/presigned-read-url",
  "strategy": "hierarchical",
  "chunk_size": 500,
  "chunk_overlap": 50
}
```

Exactly one of `text`, `file_url`, or `read_url` must be provided.

Response body (`ChunkerPredictResponse`):

```json
{
  "chunks": [
    {
      "chunk_index": 0,
      "content": "The extracted text chunk.",
      "metadata": {"source": "https://example.com/doc.pdf"}
    }
  ]
}
```

---

## 7. Integration Guide

### 7.1 Direct Python Usage

Use the OCR plugin directly in your own code — no FastAPI service needed.

```python
import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling_custom_ocr.api_ocr import CustomApiOcrOptions

# 1. Configure the custom OCR options
ocr_options = CustomApiOcrOptions(
    url="https://your-ocr-service.example.com/ocr",
    lang=["en"],          # language hints for your OCR service
    timeout=120.0,        # seconds; increase for slow services
    scale=3.0,            # upscale factor — higher = better OCR, slower
    image_format="JPEG",  # "PNG" or "JPEG"
    confidence_threshold=0.0,  # drop predictions below this confidence
)

# 2. Build pipeline options
pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    allow_external_plugins=True,   # required — enables plugin discovery
    force_full_page_ocr=True,      # OCR every page, even those with a text layer
    ocr_options=ocr_options,
)

# 3. Create the converter
converter = DocumentConverter(
    format_options={
        InputFormat.PDF:   PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }
)

# 4. Convert and export
result = converter.convert("invoice.pdf")
markdown = result.document.export_to_markdown()
print(markdown)
```

**For images** (PNG, JPG, TIFF):

```python
result = converter.convert("scanned_page.png")
```

**Chunking after conversion:**

```python
from docling.chunking import HierarchicalChunker

chunker = HierarchicalChunker()
chunks = list(chunker.chunk(result.document))

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk.text)
```

**Using Surya instead of the custom API:**

```python
from docling_surya import SuryaOcrOptions

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_model="suryaocr",
    allow_external_plugins=True,
    ocr_options=SuryaOcrOptions(lang=["en", "ar"]),
)
```

### 7.2 Via the FastAPI Service (HTTP)

Start the server, then call it from any language.

```powershell
# Start the server
conda activate raas2
python src/main.py
```

**Python client:**

```python
import requests, io

# --- Raw text ---
r = requests.post(
    "http://localhost:8100/process-document/",
    data={"text": "Your text here.", "chunking_strategy": "recursive"},
)
chunks = r.json()["chunks"]

# --- Upload a PDF file ---
with open("invoice.pdf", "rb") as f:
    r = requests.post(
        "http://localhost:8100/process-document/",
        files={"file": ("invoice.pdf", f, "application/pdf")},
        data={"use_ocr": "true", "chunking_strategy": "hierarchical"},
    )
chunks = r.json()["chunks"]

# --- Gateway contract ---
r = requests.post(
    "http://localhost:8100/predict",
    json={
        "text": "Hello from the gateway.",
        "strategy": "hierarchical",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
)
chunks = r.json()["chunks"]
```

**JavaScript/Node.js client:**

```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const form = new FormData();
form.append('file', fs.createReadStream('document.pdf'));
form.append('use_ocr', 'true');
form.append('chunking_strategy', 'hierarchical');

const res = await fetch('http://localhost:8100/process-document/', {
    method: 'POST',
    body: form,
});
const { chunks } = await res.json();
```

### 7.3 Via Docker

```powershell
# Build
docker build -t docling-chunker .

# Run — all config comes from your .env file
docker run --gpus all -p 8100:8100 --env-file .env docling-chunker

# Override a single variable at run time (e.g. switch to Surya)
docker run --gpus all -p 8100:8100 --env-file .env -e OCR_BACKEND=surya docling-chunker

# CPU-only (no GPU flag)
docker run -p 8100:8100 --env-file .env -e OCR_BACKEND=surya docling-chunker
```

### 7.4 Embed in an Existing Project

To add the custom OCR plugin to any existing project:

**Step 1** — Copy the plugin package:

```
your-project/
└── src/
    └── docling-custom-ocr/      ← copy this from our repo
        ├── pyproject.toml
        └── src/docling_custom_ocr/
            ├── __init__.py
            ├── plugin.py
            └── api_ocr.py
```

**Step 2** — Install it:

```powershell
pip install -e path/to/docling-custom-ocr
```

**Step 3** — Use it (see [Section 7.1](#71-direct-python-usage)):

```python
from docling_custom_ocr.api_ocr import CustomApiOcrOptions
# … same as above
```

That is all. The plugin self-registers via the entry-point; your code only needs to set `allow_external_plugins=True` and pass a `CustomApiOcrOptions` instance.

---

## 8. OCR API Response Formats

The plugin can consume three different response shapes from your OCR service. Pick whichever fits your API.

### Format A — flat list of predictions

```json
[
  {"text": "Invoice Number", "bbox": [12, 30, 150, 48], "confidence": 0.98},
  {"text": "1042",           "bbox": [160, 30, 210, 48], "confidence": 0.97}
]
```

### Format B — flat dict (single cell)

```json
{"text": "Total: $1,200", "bbox": [10, 200, 200, 220], "confidence": 0.95}
```

### Format C — document-level JSON with `cells_data`

```json
{
  "cells_data": [
    {"text": "Word one", "bbox": [0, 0, 80, 15], "confidence": 0.9},
    {"text": "Word two", "bbox": [85, 0, 160, 15], "confidence": 0.88}
  ]
}
```

### Format D — document-level JSON with `page_results`

```json
{
  "page_results": [
    {
      "cells_data": [
        {"text": "Page 1 word", "bbox": [10, 10, 100, 25]}
      ]
    }
  ]
}
```

### Format E — nested JSON string in the `text` field

Some APIs wrap their word-level predictions as a JSON string inside the top-level `text` field:

```json
{
  "text": "[{\"text\": \"Hello\", \"bbox\": [0, 0, 60, 15]}, {\"text\": \"World\", \"bbox\": [65, 0, 130, 15]}]",
  "confidence": 0.94
}
```

The plugin automatically parses and unwraps all five formats. Bounding box values must be `[left, top, right, bottom]` in **pixel coordinates** relative to the image that was sent (before scale mapping).

---

## 9. Environment Variables

### 9.1 The `.env` file

All configuration lives in `.env` at the project root. `src/main.py` loads it automatically at startup:

```python
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
```

This means you never need to set shell environment variables manually — just edit `.env` and restart the server.

**`.env.example`** is the committed template. Copy it to `.env` and fill in real values:

```powershell
copy .env.example .env
```

`.env` is listed in `.gitignore` and must never be committed — it may contain API keys or internal URLs.

### 9.2 All variables

| Variable | Default in `.env` | Description |
|----------|-------------------|-------------|
| `OCR_BACKEND` | `custom_api` | Which OCR engine to use: `custom_api` or `surya` |
| `CUSTOM_OCR_API_URL` | *(required — set in `.env`)* | Your OCR service endpoint (only used when `OCR_BACKEND=custom_api`) |
| `CUSTOM_OCR_TIMEOUT` | `300` | Request timeout in seconds |
| `SERVICE_PORT` | `8100` | Port the FastAPI service listens on |
| `DOCLING_ALLOW_EXTERNAL_PLUGINS` | `true` | Must be `true` for plugin discovery |

### 9.3 Switching between backends

Open `.env` and change one line — no code changes needed:

```ini
# Use your custom OCR API
OCR_BACKEND=custom_api
CUSTOM_OCR_API_URL=https://your-ocr-service.example.com/ocr

# — or —

# Use Surya (local, no external dependency)
OCR_BACKEND=surya
```

Restart the server after any change.

### 9.4 How values flow into the code

When `OCR_BACKEND=custom_api`, `extract_with_ocr()` builds:

```python
CustomApiOcrOptions(
    lang=["ar", "en"],
    url=os.getenv("CUSTOM_OCR_API_URL"),           # required — must be set in .env
    timeout=float(os.getenv("CUSTOM_OCR_TIMEOUT", "300")),
)
```

`CUSTOM_OCR_API_URL` has no hardcoded fallback. If it is missing from `.env`, Pydantic raises a `ValidationError: url field required` immediately at startup — a clear signal to check your `.env` rather than silently hitting the wrong endpoint.

When `OCR_BACKEND=surya`, it builds `SuryaOcrOptions` instead — no HTTP call to any custom endpoint is made.

### 9.5 Precedence

`python-dotenv` only sets a variable if it is **not already set** in the shell environment. Shell variables always win:

```powershell
# This overrides whatever is in .env for this one run
$env:OCR_BACKEND = "surya"
python src/main.py
```

### 9.6 Docker

Pass the `.env` file directly to the container — no need to repeat each variable individually:

```powershell
docker run --gpus all -p 8100:8100 --env-file .env docling-chunker
```

Or override specific values at run time:

```powershell
docker run --gpus all -p 8100:8100 `
  --env-file .env `
  -e OCR_BACKEND=surya `
  docling-chunker
```

---

## 10. Testing

Start the server in one terminal before running any server-dependent tests:

```powershell
conda activate raas2
python src/main.py
```

---

### 10.1 Swagger UI — interactive file upload (browser)

Open **http://localhost:8100/docs**

FastAPI generates this automatically — no extra setup. Every endpoint has a **Try it out** button with a form. For `/process-document/` you get a file picker, a `use_ocr` toggle, and a `chunking_strategy` dropdown. Use this for one-off manual tests with any file on your machine.

---

### 10.2 `test_upload.py` — terminal script

```powershell
# Run all built-in demos (raw text + sample files)
python test_upload.py

# Upload any file from disk
python test_upload.py path\to\document.pdf

# Force a specific backend for this run
python test_upload.py path\to\scan.pdf surya
```

Output per request: HTTP status, chunk count, and a 120-character preview of the first chunk. Sample files in `tests/samples/` are tested automatically if present; missing files are skipped with `[SKIP]`.

To add sample files:

```
tests/samples/
├── sample_text.pdf     ← PDF with embedded text layer
├── sample_scan.pdf     ← scanned/image-only PDF
├── sample.png          ← scanned image
├── sample.docx         ← Word document
└── sample.txt          ← plain text
```

---

### 10.3 Unit tests — fast, no server, all deps mocked

```powershell
pytest tests/test_main.py -v
```

Covers all API logic, routing, chunking strategies, and validation without touching Docling, Surya, or any real OCR endpoint.

---

### 10.4 Plugin tests — no server needed

```powershell
pip install -e src/custom-ocr   # once
pytest tests/test_integration.py -v -m plugin
```

Covers: `CustomApiOcrOptions` validation, `_get_confidence_value`, `_map_bbox_to_page`, `_iter_prediction_candidates`, `_extract_cells_from_document_payload`, mocked HTTP call, plugin registration hook.

---

### 10.5 API tests — requires running server

```powershell
pytest tests/test_integration.py -v -m api
```

Covers: health endpoint, raw text chunking, all five `ChunkingStrategy` enum values, input validation errors, `/predict` endpoint, and file upload routing (uses sample files if present).

---

### 10.6 Full pipeline test — real Docling inference (slow)

```powershell
pytest tests/test_integration.py -v -m slow
```

Runs a complete Docling conversion on `tests/samples/sample_scan.pdf` with the HTTP OCR call mocked. Requires `tests/samples/sample_scan.pdf` to exist.

---

### 10.7 Quick smoke test — plain script, no pytest

```powershell
python tests/test_integration.py
```

Runs plugin checks unconditionally and hits the live API if the server is up.

---

## 11. Common Pitfalls

### Config changes in `.env` have no effect

The server reads `.env` once at startup via `load_dotenv`. You must **restart the server** after editing `.env` — the running process does not watch for changes.

Also confirm `.env` exists at the project root (not inside `src/`). If it is missing, `load_dotenv` silently does nothing and `CUSTOM_OCR_API_URL` will be `None` — Pydantic will raise `ValidationError: url field required` at the first OCR request, which is the expected signal to check your `.env`.

### `ModuleNotFoundError: No module named 'docling_custom_ocr'`

The plugin is not installed. Run:

```powershell
pip install -e src/custom-ocr
```

### `KeyError` or OCR falls back to default engine silently

`allow_external_plugins=True` is missing from `PdfPipelineOptions`. The factory only discovers external plugins when this flag is set.

### `TypeError: __init__() got an unexpected keyword argument 'enabled'`

Your `CustomApiOcrModel.__init__` does not accept `enabled`. The factory always passes it. Fix:

```python
def __init__(self, enabled: bool, artifacts_path, options, accelerator_options):
    super().__init__(enabled=enabled, ...)
```

### `ValidationError` on `CustomApiOcrOptions`

Either `url` is missing (required field), `lang` is missing (required by base class), or `extra="forbid"` is catching a typo in a field name.

### `'dict' object has no attribute 'layout'`

You replaced `page.predictions` with a dict directly. Use `post_process_cells(cells, page)` instead — it is the only safe injection point.

### `pydantic.ValidationError: ... coord_origin`

Bounding boxes must be constructed with `CoordOrigin.TOPLEFT` when coordinates come from image space (top-left origin). Docling's internal coordinate system expects this origin.

### Windows: HuggingFace model download fails with symlink error

Surya downloads model weights to `~/.cache/huggingface`. On Windows, symlinks require elevated permissions. Fix by running as administrator once, or by setting:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

### OCR returns text but Markdown export is empty

`force_full_page_ocr=True` is missing. Without it, Docling only runs OCR on regions that lack a text layer. If the PDF has a text layer but it is incorrect (badly OCR'd elsewhere), the text layer wins and your OCR is ignored.

---

## 12. How It Compares to the Built-in OCR Engines

| Feature | `custom_api` (this plugin) | `suryaocr` | Tesseract |
|---------|---------------------------|-----------|-----------|
| Runs locally | No (HTTP call) | Yes (GPU/CPU) | Yes (CPU) |
| Arabic support | Depends on your service | Yes | Yes (ara model) |
| GPU required | No | Recommended | No |
| Cold start | Fast | Slow (model load) | Fast |
| Accuracy | Depends on your service | High | Moderate |
| Custom models | Yes — you control the API | No | No |
| Rate limiting | Depends on your service | No | No |

---

**This README reflects the actual implementation in `src/custom-ocr/src/docling_custom_ocr/api_ocr.py` and `src/main.py` as of the current commit.**
