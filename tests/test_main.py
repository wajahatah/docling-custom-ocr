"""
Unit Tests for RAG Document Processing API
Run with: pytest tests/test_main.py -v
"""

import io

# ---------------------------------------------------------------------------
# We mock heavy dependencies before importing the app so tests run fast
# without needing Docling/Surya/GPU installed in the test environment.
# ---------------------------------------------------------------------------
# Mock docling_surya before it's imported by main.py
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

mock_surya = MagicMock()
mock_surya.SuryaOcrOptions = MagicMock(return_value=MagicMock())
sys.modules["docling_surya"] = mock_surya

# Mock langchain_text_splitters so tests don't need the package installed
mock_langchain = MagicMock()
mock_splitter_instance = MagicMock()
mock_splitter_instance.split_text.return_value = ["chunk one", "chunk two", "chunk three"]
mock_langchain.RecursiveCharacterTextSplitter.return_value = mock_splitter_instance
sys.modules["langchain_text_splitters"] = mock_langchain

from src.main import app, chunk_document, chunk_raw_text  # noqa: E402

client = TestClient(app)


# ===========================================================================
# Helpers
# ===========================================================================

def make_fake_conversion_result(texts: list[str]):
    """
    Build a minimal mock ConversionResult whose .document can be passed
    into chunk_document without touching real Docling internals.
    """
    mock_meta = MagicMock()
    mock_meta.export_json_dict.return_value = {"source": "mock"}

    mock_chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.text = text
        chunk.meta = mock_meta
        mock_chunks.append(chunk)

    mock_doc = MagicMock()
    mock_result = MagicMock()
    mock_result.document = mock_doc
    return mock_result, mock_chunks


def make_upload_file(filename: str, content: bytes = b"fake content"):
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


# ===========================================================================
# 1. Health Check
# ===========================================================================

class TestHealthCheck:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# ===========================================================================
# 2. Input Validation
# ===========================================================================

class TestInputValidation:

    def test_no_input_returns_400(self):
        """Neither file nor text provided."""
        response = client.post("/process-document/")
        assert response.status_code == 400
        assert "either" in response.json()["detail"].lower()

    def test_both_inputs_returns_400(self):
        """Both file and text provided simultaneously."""
        response = client.post(
            "/process-document/",
            data={"text": "some text"},
            files=make_upload_file("test.pdf"),
        )
        assert response.status_code == 400
        assert "not both" in response.json()["detail"].lower()

    def test_unsupported_file_extension_returns_400(self):
        """File with unsupported extension."""
        response = client.post(
            "/process-document/",
            files=make_upload_file("malware.exe"),
        )
        assert response.status_code == 400
        assert "unsupported file type" in response.json()["detail"].lower()

    def test_unsupported_extension_lists_accepted_formats(self):
        """Error message should tell user what IS accepted."""
        response = client.post(
            "/process-document/",
            files=make_upload_file("file.xyz"),
        )
        assert ".pdf" in response.json()["detail"]


# ===========================================================================
# 3. Raw Text Path
# ===========================================================================

class TestRawTextProcessing:

    def test_raw_text_returns_200(self):
        response = client.post(
            "/process-document/",
            data={"text": "Hello world. This is a test sentence."},
        )
        assert response.status_code == 200

    def test_raw_text_response_structure(self):
        response = client.post(
            "/process-document/",
            data={"text": "Hello world. This is a test."},
        )
        body = response.json()
        assert body["status"] == "success"
        assert body["source"] == "raw_text"
        assert "total_chunks" in body
        assert "chunks" in body
        assert isinstance(body["chunks"], list)

    def test_raw_text_chunk_structure(self):
        """Each chunk must have chunk_index, text, and meta fields."""
        response = client.post(
            "/process-document/",
            data={"text": "Hello world. This is a test."},
        )
        chunk = response.json()["chunks"][0]
        assert "chunk_index" in chunk
        assert "text" in chunk
        assert "meta" in chunk

    def test_raw_text_meta_source_is_raw_text(self):
        response = client.post(
            "/process-document/",
            data={"text": "Hello world. This is a test."},
        )
        chunk = response.json()["chunks"][0]
        assert chunk["meta"]["source"] == "raw_text"

    def test_raw_text_default_chunking_strategy(self):
        response = client.post(
            "/process-document/",
            data={"text": "Hello world. This is a test."},
        )
        assert response.json()["chunking_strategy"] == "hierarchical"

    def test_raw_text_unsupported_strategy_returns_400(self):
        response = client.post(
            "/process-document/",
            data={"text": "Hello world.", "chunking_strategy": "semantic"},
        )
        assert response.status_code == 400
        assert "unsupported chunking strategy" in response.json()["detail"].lower()

    def test_raw_text_recursive_strategy_returns_200(self):
        response = client.post(
            "/process-document/",
            data={"text": "Hello world. This is a test.", "chunking_strategy": "recursive"},
        )
        assert response.status_code == 200
        assert response.json()["chunking_strategy"] == "recursive"

    def test_whitespace_only_text_treated_as_empty(self):
        """Whitespace-only text should be treated as no input → 400."""
        response = client.post(
            "/process-document/",
            data={"text": "   "},
        )
        assert response.status_code == 400


# ===========================================================================
# 4. chunk_raw_text() Unit Tests
# ===========================================================================

class TestChunkRawText:

    def test_short_text_produces_one_chunk(self):
        chunks = chunk_raw_text("Hello world.", chunk_size=500)
        assert len(chunks) == 1

    def test_chunk_index_is_sequential(self):
        long_text = ("This is sentence number one. " * 30).strip()
        chunks = chunk_raw_text(long_text, chunk_size=100)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_all_chunks_have_required_fields(self):
        chunks = chunk_raw_text("Hello world. Goodbye world.")
        for chunk in chunks:
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert "meta" in chunk

    def test_no_empty_chunks(self):
        chunks = chunk_raw_text("Hello world. This is a longer test sentence here.")
        for chunk in chunks:
            assert chunk["text"].strip() != ""

    def test_long_text_produces_multiple_chunks(self):
        # Use punctuated sentences so the sentence splitter triggers correctly
        long_text = ("This is a test sentence. " * 30).strip()
        chunks = chunk_raw_text(long_text, chunk_size=100)
        assert len(chunks) > 1

    def test_long_text_no_punctuation_produces_multiple_chunks(self):
        # Text with no sentence boundaries should fall back to word-level splitting
        long_text = ("word " * 200).strip()
        chunks = chunk_raw_text(long_text, chunk_size=100)
        assert len(chunks) > 1

    def test_unsupported_strategy_raises_http_exception(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            chunk_raw_text("Hello world.", strategy="unsupported")
        assert exc_info.value.status_code == 400

    def test_overlap_carries_context(self):
        """With overlap, the start of chunk N+1 should contain tail of chunk N."""
        long_text = ("This is a test sentence. " * 30).strip()
        chunks = chunk_raw_text(long_text, chunk_size=100, overlap=20)
        if len(chunks) > 1:
            tail_of_first = chunks[0]["text"][-20:]
            # The overlap tail should appear somewhere near the start of chunk 2
            assert tail_of_first.strip() in chunks[1]["text"]


# ===========================================================================
# 5. chunk_document() Unit Tests
# ===========================================================================

class TestChunkDocument:

    def test_unsupported_strategy_raises_http_exception(self):
        from fastapi import HTTPException
        mock_result, _ = make_fake_conversion_result(["text"])
        with pytest.raises(HTTPException) as exc_info:
            chunk_document(mock_result, strategy="unknown")
        assert exc_info.value.status_code == 400

    def test_supported_strategies_listed_in_error(self):
        from fastapi import HTTPException
        mock_result, _ = make_fake_conversion_result(["text"])
        with pytest.raises(HTTPException) as exc_info:
            chunk_document(mock_result, strategy="unknown")
        assert "hierarchical" in exc_info.value.detail
        assert "recursive" in exc_info.value.detail

    @patch("src.main.HierarchicalChunker")
    def test_hierarchical_strategy_returns_chunks(self, mock_chunker_class):
        mock_result, mock_chunks = make_fake_conversion_result(["chunk one", "chunk two"])
        mock_chunker_class.return_value.chunk.return_value = mock_chunks

        result = chunk_document(mock_result, strategy="hierarchical")

        assert len(result) == 2
        assert result[0]["text"] == "chunk one"
        assert result[1]["text"] == "chunk two"

    @patch("src.main.HierarchicalChunker")
    def test_chunk_indices_are_sequential(self, mock_chunker_class):
        mock_result, mock_chunks = make_fake_conversion_result(["a", "b", "c"])
        mock_chunker_class.return_value.chunk.return_value = mock_chunks

        result = chunk_document(mock_result, strategy="hierarchical")
        assert [c["chunk_index"] for c in result] == [0, 1, 2]

    @patch("src.main.HierarchicalChunker")
    def test_meta_exported_when_available(self, mock_chunker_class):
        mock_result, mock_chunks = make_fake_conversion_result(["text"])
        mock_chunker_class.return_value.chunk.return_value = mock_chunks

        result = chunk_document(mock_result, strategy="hierarchical")
        assert result[0]["meta"] == {"source": "mock"}

    def test_recursive_strategy_returns_chunks(self):
        """Recursive strategy exports markdown then splits with LangChain."""
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "some long document text"

        result = chunk_document(mock_result, strategy="recursive", chunk_size=100, overlap=20)

        assert len(result) > 0
        assert all("chunk_index" in c for c in result)
        assert all("text" in c for c in result)
        assert all(c["meta"]["source"] == "recursive" for c in result)

    def test_recursive_strategy_calls_export_to_markdown(self):
        """Recursive strategy must export the Docling document to text first."""
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "document text"

        chunk_document(mock_result, strategy="recursive", chunk_size=100, overlap=20)
        mock_result.document.export_to_markdown.assert_called_once()

    def test_recursive_strategy_indices_are_sequential(self):
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "document text"
        result = chunk_document(mock_result, strategy="recursive", chunk_size=100, overlap=0)
        assert [c["chunk_index"] for c in result] == list(range(len(result)))


# ===========================================================================
# 6. File Path — Mocked Docling Extraction
# ===========================================================================

class TestFileProcessing:

    @patch("src.main.extract_with_ocr")
    @patch("src.main.chunk_document")
    def test_pdf_routes_to_ocr_extractor(self, mock_chunk, mock_extract):
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            files=make_upload_file("test.pdf"),
        )
        assert response.status_code == 200
        mock_extract.assert_called_once()

    @patch("src.main.standardDocling")
    @patch("src.main.chunk_document")
    def test_docx_routes_to_standard_extractor(self, mock_chunk, mock_standard):
        mock_standard.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            files=make_upload_file("test.docx"),
        )
        assert response.status_code == 200
        mock_standard.assert_called_once()

    @patch("src.main.extract_with_ocr")
    @patch("src.main.chunk_document")
    def test_file_response_includes_filename(self, mock_chunk, mock_extract):
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            files=make_upload_file("my_document.pdf"),
        )
        assert response.json()["filename"] == "my_document.pdf"

    @patch("src.main.extract_with_ocr")
    @patch("src.main.chunk_document")
    def test_file_response_source_is_file(self, mock_chunk, mock_extract):
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            files=make_upload_file("test.pdf"),
        )
        assert response.json()["source"] == "file"

    @patch("src.main.extract_with_ocr")
    @patch("src.main.chunk_document")
    def test_image_routes_to_ocr_extractor(self, mock_chunk, mock_extract):
        """PNG files should always go through OCR regardless of use_ocr flag."""
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            files=make_upload_file("scan.png"),
        )
        assert response.status_code == 200
        mock_extract.assert_called_once()

    @patch("src.main.extract_with_ocr")
    @patch("src.main.chunk_document")
    def test_image_ignores_use_ocr_false(self, mock_chunk, mock_extract):
        """Images always use OCR — use_ocr=False should have no effect."""
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            data={"use_ocr": "false"},
            files=make_upload_file("scan.jpg"),
        )
        assert response.status_code == 200
        mock_extract.assert_called_once()

    @patch("src.main.extract_with_ocr")
    @patch("src.main.chunk_document")
    def test_pdf_use_ocr_true_routes_to_extract_with_ocr(self, mock_chunk, mock_extract):
        """PDF + use_ocr=True (default) → extract_with_ocr."""
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            data={"use_ocr": "true"},
            files=make_upload_file("test.pdf"),
        )
        assert response.status_code == 200
        assert response.json()["use_ocr"] is True
        mock_extract.assert_called_once()

    @patch("src.main.extract_pdf_without_ocr")
    @patch("src.main.chunk_document")
    def test_pdf_use_ocr_false_routes_to_extract_pdf_without_ocr(self, mock_chunk, mock_extract):
        """PDF + use_ocr=False → extract_pdf_without_ocr."""
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            data={"use_ocr": "false"},
            files=make_upload_file("test.pdf"),
        )
        assert response.status_code == 200
        assert response.json()["use_ocr"] is False
        mock_extract.assert_called_once()

    @patch("src.main.standardDocling")
    @patch("src.main.chunk_document")
    def test_docx_ignores_use_ocr_flag(self, mock_chunk, mock_standard):
        """Standard docs always go to standardDocling regardless of use_ocr."""
        mock_standard.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            data={"use_ocr": "false"},
            files=make_upload_file("test.docx"),
        )
        assert response.status_code == 200
        mock_standard.assert_called_once()

    @patch("src.main.extract_with_ocr")
    @patch("src.main.chunk_document")
    def test_use_ocr_echoed_in_response(self, mock_chunk, mock_extract):
        """use_ocr value should appear in the response body."""
        mock_extract.return_value = MagicMock()
        mock_chunk.return_value = [{"chunk_index": 0, "text": "hi", "meta": {}}]

        response = client.post(
            "/process-document/",
            files=make_upload_file("test.pdf"),
        )
        assert "use_ocr" in response.json()


# ===========================================================================
# 7. extract_pdf_without_ocr unit tests
# ===========================================================================

class TestExtractPdfWithoutOcr:

    @patch("src.main.PdfFormatOption")
    @patch("src.main.PdfPipelineOptions")
    @patch("src.main.DocumentConverter")
    def test_converter_created_with_do_ocr_false(
        self, mock_converter_class, mock_pipeline, mock_format_option
    ):
        """Verify PdfPipelineOptions is configured with do_ocr=False."""
        from src.main import extract_pdf_without_ocr

        mock_instance = MagicMock()
        mock_instance.convert.return_value = MagicMock()
        mock_converter_class.return_value = mock_instance
        mock_pipeline.return_value = MagicMock()
        mock_format_option.return_value = MagicMock()

        extract_pdf_without_ocr("/fake/path.pdf")
        mock_pipeline.assert_called_once_with(do_ocr=False)

    @patch("src.main.DocumentConverter")
    def test_converter_is_called_with_file_path(self, mock_converter_class):
        """Converter should call .convert() with the provided path."""
        from src.main import extract_pdf_without_ocr

        mock_instance = MagicMock()
        mock_instance.convert.return_value = MagicMock()
        mock_converter_class.return_value = mock_instance

        extract_pdf_without_ocr("/fake/path.pdf")
        mock_instance.convert.assert_called_once_with("/fake/path.pdf")

    @patch("src.main.DocumentConverter")
    def test_returns_conversion_result(self, mock_converter_class):
        """Function should return the result of converter.convert()."""
        from src.main import extract_pdf_without_ocr

        fake_result = MagicMock()
        mock_instance = MagicMock()
        mock_instance.convert.return_value = fake_result
        mock_converter_class.return_value = mock_instance

        result = extract_pdf_without_ocr("/fake/path.pdf")
        assert result is fake_result


# ===========================================================================
# 8. Recursive Chunking — chunk_raw_text() with strategy="recursive"
#    Uses LangChain RecursiveCharacterTextSplitter under the hood.
# ===========================================================================

class TestRecursiveChunking:

    def test_recursive_returns_chunks(self):
        chunks = chunk_raw_text("Hello world.\n\nThis is paragraph two.", strategy="recursive")
        assert len(chunks) > 0

    def test_recursive_chunk_structure(self):
        chunks = chunk_raw_text("Hello world. Goodbye world.", strategy="recursive")
        for chunk in chunks:
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert "meta" in chunk

    def test_recursive_meta_source(self):
        chunks = chunk_raw_text("Hello world.", strategy="recursive")
        assert chunks[0]["meta"]["source"] == "raw_text"

    def test_recursive_indices_are_sequential(self):
        chunks = chunk_raw_text("word " * 300, strategy="recursive", chunk_size=100)
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_recursive_no_empty_chunks(self):
        chunks = chunk_raw_text("sentence. " * 50, strategy="recursive", chunk_size=100)
        for chunk in chunks:
            assert chunk["text"].strip() != ""

    def test_recursive_uses_langchain_splitter(self):
        """Verify RecursiveCharacterTextSplitter is invoked with correct params."""
        with patch("src.main.RecursiveCharacterTextSplitter") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.split_text.return_value = ["chunk one", "chunk two"]
            mock_cls.return_value = mock_instance

            chunk_raw_text("some text", strategy="recursive", chunk_size=200, overlap=30)

            mock_cls.assert_called_once_with(
                chunk_size=200,
                chunk_overlap=30,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
            mock_instance.split_text.assert_called_once_with("some text")

    def test_recursive_chunk_size_passed_to_langchain(self):
        """chunk_size parameter must be forwarded to LangChain."""
        with patch("src.main.RecursiveCharacterTextSplitter") as mock_cls:
            mock_cls.return_value.split_text.return_value = ["a"]
            chunk_raw_text("text", strategy="recursive", chunk_size=999, overlap=10)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["chunk_size"] == 999

    def test_recursive_overlap_passed_to_langchain(self):
        """overlap parameter must be forwarded as chunk_overlap to LangChain."""
        with patch("src.main.RecursiveCharacterTextSplitter") as mock_cls:
            mock_cls.return_value.split_text.return_value = ["a"]
            chunk_raw_text("text", strategy="recursive", chunk_size=200, overlap=75)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["chunk_overlap"] == 75

    def test_recursive_unsupported_strategy_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            chunk_raw_text("Hello.", strategy="semantic")
        assert exc_info.value.status_code == 400
        assert "hierarchical" in exc_info.value.detail
        assert "recursive" in exc_info.value.detail

    def test_recursive_via_api_endpoint(self):
        response = client.post(
            "/process-document/",
            data={
                "text": "Paragraph one.\n\nParagraph two.\n\nParagraph three.",
                "chunking_strategy": "recursive",
            },
        )
        assert response.status_code == 200
        assert response.json()["chunking_strategy"] == "recursive"
        assert len(response.json()["chunks"]) > 0
