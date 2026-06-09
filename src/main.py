"""
RAG Document Processing API
FastAPI wrapper around the Docling modular pipeline.

Two API surfaces:
  - `POST /process-document/` — multipart upload, the legacy endpoint.
    Useful for direct browser/curl upload tests.
  - `POST /predict` — JSON, called by the AI Gateway. Accepts a
    ChunkerPredictRequest (text | file_url | read_url) and returns a
    ChunkerPredictResponse. The URL fetch + extraction + chunking
    pipeline live here so the AI Gateway stays a thin HTTP proxy.
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load .env from the project root (one level above src/)
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx
from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling_surya import SuryaOcrOptions
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importing the gateway contract so /predict accepts/returns the same
# shape every service in the platform agrees on. The repo root is added
# to sys.path by the chunking service's container/launch script.
from contracts.contracts import(
    ChunkerPredictRequest,
    ChunkerPredictResponse,
)
from contracts.schemas import ChunkingStrategy

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Document Processing API")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS     = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
PDF_EXTENSIONS       = {".pdf"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | {".docx", ".txt", ".pptx", ".xlsx", ".html", ".md"}
SUPPORTED_STRATEGIES = {"hierarchical", "recursive"}

CHUNKS_DIR = Path(__file__).parent.parent / "chunks"


def _save_chunks(stem: str, chunks: list) -> Path:
    """Write chunks to chunks/{stem}_chunks.json at the project root."""
    CHUNKS_DIR.mkdir(exist_ok=True)
    out = CHUNKS_DIR / f"{stem}_chunks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[Chunks] Saved {len(chunks)} chunk(s) → {out}")
    return out

# ---------------------------------------------------------------------------
# Extraction Function A — With OCR (PDF & Images)
# ---------------------------------------------------------------------------

def extract_with_ocr(file_path: str):
    ocr_backend = os.getenv("OCR_BACKEND", "custom_api").strip().lower()
    print(f"[OCR] Processing '{file_path}' with {ocr_backend} (ar + en)...")


    if ocr_backend == "custom_api":
        try:
            from docling_custom_ocr.api_ocr import (
                CustomApiOcrOptions,  # type: ignore[reportMissingImports]
            )
        except Exception as exc:
            raise RuntimeError(
                "OCR_BACKEND=custom_api requires the docling-custom-ocr plugin to be installed. "
                "Install it with: pip install -e ./src/docling-custom-ocr"
            ) from exc

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            allow_external_plugins=True,
            ocr_options=CustomApiOcrOptions(
                lang=["ar","en"],
                url=os.getenv("CUSTOM_OCR_API_URL"),
                timeout=float(os.getenv("CUSTOM_OCR_TIMEOUT", "300")),
            ),
            force_full_page_ocr=True,

        )
    else:
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_model="suryaocr",
            allow_external_plugins=True,
            ocr_options=SuryaOcrOptions(lang=["en", "ar"]),
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter.convert(file_path)

# ---------------------------------------------------------------------------
# Extraction Function B — Standard Docling (docx, txt, pptx, etc.)
# Uses Docling's native parser for each format.
# NOTE: OCR on embedded images inside .docx requires LibreOffice installed
# and available in PATH (or via DOCLING_LIBREOFFICE_CMD env variable).
# PdfPipelineOptions are intentionally NOT applied here — they are PDF-specific
# and are silently ignored by Docling's DOCX/PPTX/TXT parsers.
# ---------------------------------------------------------------------------

def standardDocling(file_path: str):
    print(f"[Standard] Processing '{file_path}' with native Docling parser...")

    converter = DocumentConverter()
    return converter.convert(file_path)

# ---------------------------------------------------------------------------
# Extraction Function C — PDF without OCR
# Use when the PDF contains a proper text layer (i.e. it is not a scanned
# image-only PDF) and you want to skip the Surya OCR engine entirely for
# faster processing. Layout analysis and table detection still run.
# ---------------------------------------------------------------------------

def extract_pdf_without_ocr(file_path: str):
    print(f"[PDF/No-OCR] Processing '{file_path}' — skipping OCR engine...")

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter.convert(file_path)

# ---------------------------------------------------------------------------
# LangChain Recursive Splitter — shared helper used by both chunk_document
# and chunk_raw_text when strategy == "recursive".
#
# RecursiveCharacterTextSplitter tries separators in order:
#   "\n\n" (paragraph) → "\n" (line) → " " (word) → "" (character)
# until every chunk fits within chunk_size. Overlap is applied at word
# boundaries so chunks never cut mid-word.
# ---------------------------------------------------------------------------

def _langchain_recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Chunking Function — for Docling ConversionResult (file path)
# ---------------------------------------------------------------------------

def chunk_document(
    conversion_result,
    strategy: str = "hierarchical",
    chunk_size: int = 500,
    overlap: int = 50,
    source: Optional[str] = None,
) -> list[dict]:
    print(f"[Chunking] Strategy: '{strategy}'...")

    if strategy not in SUPPORTED_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported chunking strategy '{strategy}'. "
                f"Currently supported: {sorted(SUPPORTED_STRATEGIES)}."
            ),
        )

    if strategy == "hierarchical":
        chunker = HierarchicalChunker()
        chunks = list(chunker.chunk(conversion_result.document))

        serialised = []
        for i, chunk in enumerate(chunks):
            serialised.append({
                "chunk_index": i,
                "text": chunk.text,
                "meta": chunk.meta.export_json_dict() if hasattr(chunk.meta, "export_json_dict") else {},
            })

    elif strategy == "recursive":
        # Export the full document text then split with LangChain
        full_text = conversion_result.document.export_to_markdown()
        pieces = _langchain_recursive_split(full_text, chunk_size=chunk_size, overlap=overlap)
        serialised = [
            {"chunk_index": i, "text": piece, "meta": {"source": source, "strategy": "recursive"}}
            for i, piece in enumerate(pieces)
        ]

    print(f"[Chunking] Produced {len(serialised)} chunk(s).")
    return serialised

# ---------------------------------------------------------------------------
# Raw Text Chunking — sentence-aware splitter
# Used when the user submits plain text directly, bypassing Docling entirely.
# HierarchicalChunker requires a DoclingDocument, so raw text uses this
# lightweight fallback instead of a brittle mock object.
# ---------------------------------------------------------------------------

def chunk_raw_text(
    text: str,
    strategy: str = "hierarchical",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Split a raw string into chunks using the requested strategy.

    - hierarchical : sentence-aware splitter with word-level fallback.
                     Only valid for raw text — file uploads should use
                     Docling's HierarchicalChunker via chunk_document().
    - recursive    : LangChain RecursiveCharacterTextSplitter — tries
                     progressively finer boundaries (paragraph → line →
                     word → character) with clean word-boundary overlap.

    Args:
        text:       The raw input string.
        strategy:   'hierarchical' or 'recursive'.
        chunk_size: Target character length per chunk. Default: 500.
        overlap:    Characters of overlap between consecutive chunks. Default: 50.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported chunking strategy '{strategy}'. "
                f"Currently supported: {sorted(SUPPORTED_STRATEGIES)}."
            ),
        )

    print(f"[Chunking/Text] Strategy '{strategy}' (chunk_size={chunk_size}, overlap={overlap})...")

    if strategy == "recursive":
        pieces = _langchain_recursive_split(text, chunk_size=chunk_size, overlap=overlap)
        serialised = [
            {"chunk_index": i, "text": piece, "meta": {"source": "raw_text"}}
            for i, piece in enumerate(pieces)
        ]
        print(f"[Chunking/Text] Produced {len(serialised)} chunk(s).")
        return serialised

    # --- hierarchical path ---------------------------------------------------
    # Split on sentence boundaries (. ! ?) while preserving delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Fallback: if no sentence boundaries found (or single long sentence),
    # split by words so chunk_size is always respected.
    if len(sentences) == 1 and len(text) > chunk_size:
        words = text.split()
        sentences = []
        current_sentence = ""
        for word in words:
            if len(current_sentence) + len(word) + 1 <= chunk_size:
                current_sentence += ("" if not current_sentence else " ") + word
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = word
        if current_sentence:
            sentences.append(current_sentence)

    raw_chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += ("" if not current else " ") + sentence
        else:
            if current:
                raw_chunks.append(current)
            # Carry over trailing characters from previous chunk for context
            tail = current[-overlap:] if overlap and current else ""
            current = (tail + " " + sentence).strip()

    if current:  # flush the last chunk
        raw_chunks.append(current)

    serialised = [
        {"chunk_index": i, "text": chunk.strip(), "meta": {"source": "raw_text"}}
        for i, chunk in enumerate(raw_chunks)
    ]

    print(f"[Chunking/Text] Produced {len(serialised)} chunk(s).")
    return serialised

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Simple liveness probe — useful for deployment monitoring."""
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------------

@app.post("/process-document/")
async def process_document(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str]        = Form(default=None),
    chunking_strategy: str     = Form(default="hierarchical"),
    use_ocr: bool              = Form(default=True),
):
    """
    Upload a document OR submit raw text and receive JSON chunks.

    Exactly one of `file` or `text` must be provided per request.

    Supported file formats:
      - Images  : .png, .jpg, .jpeg, .tiff, .tif  — always OCR
      - PDF     : .pdf                             — OCR controlled by `use_ocr`
      - Standard: .docx, .txt, .pptx, .xlsx, .html, .md

    Form parameters:
      - file              : The document to upload (optional).
      - text              : Raw text string to chunk directly (optional).
      - chunking_strategy : Chunking method to apply. Default: "hierarchical".
      - use_ocr           : Apply Surya OCR when processing PDFs. Default: True.
                            Has no effect on images (always OCR) or standard docs.
    """

    # ------------------------------------------------------------------
    # 1. Input validation — exactly one of file or text must be provided
    # ------------------------------------------------------------------
    file_provided = file is not None and file.filename
    text_provided = text is not None and text.strip()

    if file_provided and text_provided:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'file' or 'text', not both.",
        )
    if not file_provided and not text_provided:
        raise HTTPException(
            status_code=400,
            detail="You must provide either a 'file' upload or a 'text' string.",
        )

    # ------------------------------------------------------------------
    # 2. Raw text path — bypass Docling entirely
    # ------------------------------------------------------------------
    if text_provided:
        print(f"[Router] Raw text input received ({len(text)} characters).")
        chunks = chunk_raw_text(text, strategy=chunking_strategy)
        _save_chunks("raw_text", chunks)
        return {
            "status": "success",
            "source": "raw_text",
            "chunking_strategy": chunking_strategy,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }

    # ------------------------------------------------------------------
    # 3. File path — validate extension, extract via Docling, then chunk
    # ------------------------------------------------------------------
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    tmp_fd, temp_file_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(tmp_fd, "wb") as tmp:
            shutil.copyfileobj(file.file, tmp)

        loop = asyncio.get_event_loop()

        if ext in IMAGE_EXTENSIONS:
            # Images always require OCR — use_ocr flag has no effect here
            print("[Router] Image detected → extract_with_ocr")
            result = await loop.run_in_executor(None, extract_with_ocr, temp_file_path)

        elif ext in PDF_EXTENSIONS:
            if use_ocr:
                print("[Router] PDF + use_ocr=True → extract_with_ocr")
                result = await loop.run_in_executor(None, extract_with_ocr, temp_file_path)
            else:
                print("[Router] PDF + use_ocr=False → extract_pdf_without_ocr")
                result = await loop.run_in_executor(None, extract_pdf_without_ocr, temp_file_path)

        else:
            # Standard documents (.docx, .txt, .pptx, etc.)
            print("[Router] Standard doc → standardDocling")
            result = await loop.run_in_executor(None, standardDocling, temp_file_path)

        chunks = await loop.run_in_executor(
            None, chunk_document, result, chunking_strategy, 500, 50
        )

        _save_chunks(Path(file.filename).stem, chunks)

        return {
            "status": "success",
            "source": "file",
            "filename": file.filename,
            "chunking_strategy": chunking_strategy,
            "use_ocr": use_ocr,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[Cleanup] Removed temporary file: {temp_file_path}")

# ---------------------------------------------------------------------------
# JSON /predict — the contract the AI Gateway calls
# ---------------------------------------------------------------------------
#
# Why this lives here and not in the AI Gateway:
#   The chunker is the only place with docling + surya + torch installed.
#   Putting URL-fetch + extraction here keeps the AI Gateway a pure HTTP
#   proxy and lets the chunker be scaled / restarted independently.
#
# Strategy mapping:
#   The shared ChunkingStrategy enum has 5 values (FIXED, RECURSIVE,
#   HIERARCHICAL, DOCUMENT_STRUCTURE, SEMANTIC). The chunker only
#   implements two boundary algorithms (recursive + hierarchical).
#   Map at the boundary so callers can use the canonical enum.

_STRATEGY_MAP: dict[ChunkingStrategy, str] = {
    ChunkingStrategy.FIXED: "recursive",
    ChunkingStrategy.RECURSIVE: "recursive",
    ChunkingStrategy.HIERARCHICAL: "hierarchical",
    ChunkingStrategy.DOCUMENT_STRUCTURE: "hierarchical",
    ChunkingStrategy.SEMANTIC: "hierarchical",
}


def _to_response_dict(raw_chunks: list[dict]) -> dict[str, Any]:
    """Reshape internal chunk dicts to ChunkerPredictResponse shape."""
    return {
        "chunks": [
            {
                "chunk_index": c["chunk_index"],
                "content": c["text"],
                "metadata": c.get("meta", {}),
            }
            for c in raw_chunks
        ]
    }


def _infer_extension(url: str, content_type: str) -> str:
    """Guess a file extension from URL path or Content-Type header."""
    ext = Path(urlparse(url).path).suffix.lower()
    if ext:
        return ext
    ct_map = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "text/plain": ".txt",
    }
    for mime, mapped_ext in ct_map.items():
        if mime in content_type:
            return mapped_ext
    return ".pdf"


def _extract_for_ext(file_path: str, ext: str):
    """Pick the right docling extraction pathway for the file extension."""
    if ext in IMAGE_EXTENSIONS or ext in PDF_EXTENSIONS:
        return extract_with_ocr(file_path)
    return standardDocling(file_path)


async def _chunk_from_url(
    url: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    source: Optional[str] = None,
) -> list[dict]:
    """Fetch a file via HTTP, extract with docling, then chunk."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("content-type", "")

    ext = _infer_extension(url, content_type)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(tmp_fd, "wb") as tmp:
            tmp.write(content)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, partial(_extract_for_ext, tmp_path, ext))
        return await loop.run_in_executor(
            None, partial(chunk_document, result, strategy, chunk_size, chunk_overlap, source)
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict", response_model=ChunkerPredictResponse)
async def predict(request: ChunkerPredictRequest) -> dict[str, Any]:
    """JSON contract: ChunkerPredictRequest -> ChunkerPredictResponse.

    Raises:
        HTTPException(400): Neither text nor a URL was provided.
        HTTPException(502): Source URL fetch or extraction failed.
    """
    strategy = _STRATEGY_MAP.get(request.strategy, "hierarchical")

    if request.text:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            partial(
                chunk_raw_text,
                request.text,
                strategy,
                request.chunk_size,
                request.chunk_overlap,
            ),
        )
        _save_chunks("predict_text", raw)
        return _to_response_dict(raw)

    source_url = request.file_url or request.read_url
    if not source_url:
        raise HTTPException(
            status_code=400,
            detail="ChunkerPredictRequest must provide text, file_url, or read_url.",
        )

    try:
        raw = await _chunk_from_url(
            source_url,
            strategy,
            request.chunk_size,
            request.chunk_overlap,
            source=request.source,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Wrap as 502 so the AI Gateway sees an upstream failure (and
        # stops retrying after the configured budget).
        raise HTTPException(status_code=502, detail=f"chunker failed: {exc}") from exc

    stem = Path(urlparse(source_url).path).stem or "predict_url"
    _save_chunks(stem, raw)
    return _to_response_dict(raw)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVICE_PORT", "8100"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
