from .api_ocr import CustomApiOcrModel


def ocr_engines():
    # Docling expects an OCR factory hook that returns available engines.
    return {"ocr_engines": [CustomApiOcrModel]}
