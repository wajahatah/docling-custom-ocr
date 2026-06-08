# Docling Custom OCR Plugin – Reference Guide

This document describes how we integrated a **custom OCR API** into **Docling (v2.70+)** using Docling’s **external plugin architecture**, without modifying Docling’s source code.

The goal is to allow Docling to route page images to our own OCR service while keeping the rest of the Docling pipeline (layout, tables, chunking, export, vectorization) unchanged.

---

## 1. High-Level Architecture

At a high level, Docling works as a **pipeline of stages**:

1. PDF parsing & page rasterization
2. OCR (optional / configurable)
3. Layout detection
4. Table / figure / equation detection
5. Assembly & export (Markdown, JSON, etc.)

Docling exposes **factories** (OCR factory, layout factory, etc.) that can dynamically load models.

We leverage this by:

* Creating a **separate Python package** (external plugin)
* Registering a **custom OCR engine**
* Implementing the OCR logic using Docling’s base classes
* Letting Docling call our OCR exactly like a built-in engine

No Docling source code is modified.

---

## 2. Key Concepts in Docling Plugin Integration

### 2.1 External Plugins

Docling supports external plugins that are:

* Installed as normal Python packages
* Discovered at runtime when `allow_external_plugins=True`

An external OCR plugin must:

* Be importable as a package
* Expose a plugin entry file (`plugin.py`)
* Return OCR model classes that extend Docling base classes

---

### 2.2 OCR Factory

Internally, Docling uses an **OCR factory** that:

* Registers available OCR engines
* Instantiates OCR models with validated options
* Calls OCR models with the signature:

```python
ocr_model(conv_res, pages)
```

Our plugin must conform to this contract.

---

## 3. Project Structure

```
docling-custom-ocr/
│
├── pyproject.toml
├── README.md
└── src/
    └── docling_custom_ocr/
        ├── __init__.py
        ├── plugin.py          # Plugin registration hook
        └── api_ocr.py         # Custom OCR implementation
```

---

## 4. Plugin Registration (`plugin.py`)

The plugin file announces OCR engines to Docling.

```python
from .api_ocr import CustomApiOcrModel

def ocr_engines():
    return {
        "ocr_engines": [CustomApiOcrModel]
    }
```

### Important Notes

* The function name and return structure must match what Docling expects.
* We return **classes**, not instances.
* Docling registers the engine under the engine’s `name` / `kind`.

---

## 5. OCR Options Model (`CustomApiOcrOptions`)

Docling uses **Pydantic models** to validate OCR configuration.

```python
class CustomApiOcrOptions(OcrOptions):
    kind: ClassVar[str] = "custom_api"

    lang: list[str] = Field(default_factory=lambda: ["en"])

    api_url: str = "https://ocrapi.llmtests.org/ocr"
    timeout_s: int = 60
```

### Key Requirements

* `kind` **must be a class-level constant**
  This is how Docling identifies the OCR engine.
* `lang` is required by Docling’s base OCR options in many versions.
* Any additional fields (API URL, timeout, auth tokens, etc.) can be added here.

---

## 6. OCR Model Implementation (`CustomApiOcrModel`)

### 6.1 Base Class

The OCR model must extend:

```python
from docling.models.base_ocr_model import BaseOcrModel
```

### 6.2 Required Class Attributes

```python
class CustomApiOcrModel(BaseOcrModel):
    name: ClassVar[str] = "custom_api"
    options_type: ClassVar[Type[OcrOptions]] = CustomApiOcrOptions
```

**Important:**

* `name` must match `CustomApiOcrOptions.kind`
* Mismatches cause subtle factory failures

---

### 6.3 Constructor Signature

Docling instantiates OCR models like this:

```python
model = OCRModel(options=..., enabled=True)
```

Therefore your constructor **must accept `enabled`** and forward it:

```python
def __init__(self, options: CustomApiOcrOptions, enabled: bool = True, **kwargs):
    super().__init__(options=options, enabled=enabled, **kwargs)
```

Failing to do this causes:

```
TypeError: __init__() got an unexpected keyword argument 'enabled'
```

---

### 6.4 OCR Execution Logic

Docling calls OCR as:

```python
ocr_model(conv_res, pages)
```

So your model **must implement**:

```python
def __call__(self, conv_res, pages) -> Iterator[Page]:
```

Key rules:

* `pages` is an iterable of `Page` objects
* You must **yield Page objects back**
* Do not return raw dicts or strings

---

### 6.5 Page → Image Conversion

In Docling 2.70, a `Page` exposes its raster image via:

```python
page.get_image()
```

We normalize it to PIL:

```python
def _page_to_pil(self, page) -> Image.Image:
    img = page.get_image()
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(np.asarray(img))
```

---

### 6.6 Calling the Custom OCR API

```python
def _call_api(self, image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    files = {"file": ("page.png", buf.getvalue(), "image/png")}
    r = requests.post(self.api_url, files=files, timeout=self.timeout_s)
    r.raise_for_status()

    data = r.json()
    if not data.get("success"):
        raise RuntimeError(data.get("error"))

    return data.get("text", "")
```

---

## 7. Injecting OCR Results Back into Docling

### 7.1 Critical Rule

**Never replace `page.predictions` with a dict.**

`page.predictions` is a Pydantic object with attributes like:

* `layout`
* `tablestructure`
* `figures_classification`

Replacing it with a dict breaks downstream stages:

```
'dict' object has no attribute 'layout'
```

---

### 7.2 Correct Update Pattern

Use `model_copy(update=...)` to preserve the predictions object:

```python
def __call__(self, conv_res, pages):
    for page in pages:
        pil_img = self._page_to_pil(page)
        text = self._call_api(pil_img)

        preds = page.predictions
        if preds is None:
            yield page
            continue

        new_preds = preds.model_copy(update={
            "vlm_response": {
                "engine": "custom_api",
                "text": text
            }
        })

        yield page.model_copy(update={"predictions": new_preds})
```

### Why This Works

* Keeps predictions as a valid Pydantic model
* Allows layout, table, and assembly stages to continue
* Docling later uses assembled text when exporting Markdown

---

## 8. Pipeline Configuration (Usage)

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True

pipeline_options.ocr_options = CustomApiOcrOptions(
    lang=["en"],
    api_url="https://ocrapi.llmtests.org/ocr",
    timeout_s=60,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert("sample.pdf")
print(result.document.export_to_markdown())
```

---

## 9. Common Pitfalls (Lessons Learned)

1. Missing `lang` → Pydantic validation error
2. Missing `enabled` in constructor → TypeError
3. Wrong `__call__` signature → OCR stage crash
4. Mutating `Page` fields directly → Pydantic error
5. Replacing `predictions` with dict → layout stage failure
6. Windows HuggingFace downloads → symlink permission issues

---

## 10. Current State

* Custom OCR plugin is fully integrated
* OCR text flows through Docling pipeline
* Markdown export includes OCR content
* No Docling source code modified

---

## 11. Next Improvements (Planned)

* Code cleanup and refactoring
* OCR result schema standardization
* Error handling & retries
* Batch OCR support
* Conditional OCR (only image pages)
* Caching & performance tuning

---

## 12. Installing the Custom Plugin

To install this custom OCR plugin into your environment, simply run `pip install` pointing to the plugin's folder (the directory that contains `pyproject.toml`).

From the parent directory of the plugin:

```bash
pip install docling-custom-ocr
```

Or from anywhere, using the full path to the plugin folder:

```bash
pip install /path/to/docling-custom-ocr
```

For development (editable install) so that code changes are picked up without reinstalling:

```bash
pip install -e docling-custom-ocr
```

Once installed, Docling will automatically discover the plugin via its entry point (`docling.ocr_engines`), and you can use it by setting `ocr_options = CustomOcrOptions(...)` as shown in the usage examples above.

---

**This README will be updated as the plugin evolves.**

---
