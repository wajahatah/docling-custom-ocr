
import json
import logging
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Literal, Optional, Tuple, Type, Union
from typing import Iterable as TypingIterable

import requests
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.models.base_ocr_model import BaseOcrModel  # Docling plugin docs mention BaseOcrModel
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from pydantic import (
    AnyUrl,
    ConfigDict,
    Field,
)


class CustomApiOcrOptions(OcrOptions):
    """Options for a custom HTTP OCR engine."""

    kind: ClassVar[Literal["custom_api"]] = "custom_api"
    lang: List[str] = ["en"]

    url: AnyUrl
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: float = 300.0
    scale: float = 3.0
    confidence_threshold: float = 0.0
    image_format: Literal["PNG", "JPEG"] = "JPEG"

    model_config = ConfigDict(
        extra="forbid",
    )


# from docling.datamodel.settings import settings
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class CustomApiOcrModel(BaseOcrModel):
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

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return CustomApiOcrOptions

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        assert self._session is not None

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: List[TextCell] = []

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue

                    high_res_image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )

                    api_predictions = self._call_api(high_res_image)
                    cells = self._build_cells(
                        predictions=api_predictions,
                        ocr_rect=ocr_rect,
                    )
                    all_ocr_cells.extend(cells)

                self.post_process_cells(all_ocr_cells, page)

            # if settings.debug.visualize_ocr:
            #     self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @staticmethod
    def _extract_cells_from_document_payload(
        payload: Dict[str, object],
    ) -> Optional[List[Dict[str, object]]]:
        """Flatten document-style OCR JSON to the list of cell dicts.

        Accepts payloads that have either:
          - top-level "cells_data": [ {text, bbox, ...}, ... ], or
          - "page_results": [ { "cells_data": [...] }, ... ]

        Each cell's "text" + "bbox" are what _iter_prediction_candidates
        already expects; any extra keys (e.g. "category") are ignored.
        Returns None if the payload does not look like this format.
        """
        collected: List[Dict[str, object]] = []
        matched = False

        page_results = payload.get("page_results")
        if isinstance(page_results, list):
            matched = True
            for pr in page_results:
                if not isinstance(pr, dict):
                    continue
                cells = pr.get("cells_data")
                if isinstance(cells, list):
                    collected.extend(c for c in cells if isinstance(c, dict))

        if not collected:
            top_cells = payload.get("cells_data")
            if isinstance(top_cells, list):
                matched = True
                collected.extend(c for c in top_cells if isinstance(c, dict))

        return collected if matched else None

    def _call_api(self, image) -> List[Dict[str, object]]:
        image_format = self.options.image_format.upper()
        image_to_save = image
        if image_format == "JPEG" and getattr(image, "mode", None) not in ("RGB", "L"):
            image_to_save = image.convert("RGB")

        buffer = BytesIO()
        image_to_save.save(buffer, format=image_format)
        buffer.seek(0)

        files = {
            "file": (
                f"page.{image_format.lower()}",
                buffer,
                f"image/{image_format.lower()}",
            )
        }

        response = None
        try:
            assert self._session is not None
            response = self._session.post(
                str(self.options.url),
                headers=self.options.headers,
                files=files,
                timeout=self.options.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            _log.error("Custom API OCR request failed: %s", exc)
            return []
        finally:
            buffer.close()

        try:
            payload = response.json()
            # with open("response.json", "w", encoding="utf-8") as f:
            #     json.dump(payload, f, indent=4)
        except ValueError:
            preview = response.text[:200]
            _log.error(
                "Custom API OCR returned invalid JSON payload: %s%s",
                preview,
                "..." if len(response.text) > len(preview) else "",
            )
            return []
        finally:
            response.close()

        if isinstance(payload, dict):
            cells = self._extract_cells_from_document_payload(payload)
            if cells is not None:
                return cells
            return [payload]
        if isinstance(payload, list):
            return [
                item for item in payload if isinstance(item, dict)
            ]

        _log.warning(
            "Unexpected payload type from Custom API OCR: %s", type(payload).__name__
        )
        return []

    def _build_cells(
        self,
        predictions: TypingIterable[Dict[str, object]],
        ocr_rect: BoundingBox,
    ) -> List[TextCell]:
        cells: List[TextCell] = []

        for prediction in predictions:
            for text, bbox, confidence in self._iter_prediction_candidates(prediction):
                if confidence < self.options.confidence_threshold:
                    continue

                mapped_bbox = self._map_bbox_to_page(
                    bbox=bbox,
                    ocr_rect=ocr_rect,
                )

                if mapped_bbox is None:
                    continue

                cells.append(
                    TextCell(
                        index=len(cells),
                        text=text,
                        orig=text,
                        from_ocr=True,
                        confidence=confidence,
                        rect=BoundingRectangle.from_bounding_box(mapped_bbox),
                    )
                )

        return cells

    def _iter_prediction_candidates(
        self, prediction: Dict[str, object]
    ) -> TypingIterable[Tuple[str, List[Union[int, float]], float]]:
        text_field = prediction.get("text")

        bbox_field = prediction.get("bbox")
        if (
            isinstance(text_field, str)
            and text_field.strip()
            and isinstance(bbox_field, (list, tuple))
            and len(bbox_field) == 4
        ):
            try:
                bbox_nums = [float(v) for v in bbox_field]
            except (TypeError, ValueError):
                pass
            else:
                confidence = self._get_confidence_value(prediction.get("confidence"))
                return [(text_field, bbox_nums, confidence)]

        if not isinstance(text_field, str):
            return []

        try:
            nested_items = json.loads(text_field)
        except json.JSONDecodeError:
            _log.debug("Failed to parse text field payload: %s", text_field)
            return []

        if isinstance(nested_items, dict):
            iterable_items: TypingIterable[Dict[str, object]] = [nested_items]
        elif isinstance(nested_items, list):
            iterable_items = [
                item for item in nested_items if isinstance(item, dict)
            ]
        else:
            return []

        default_conf = self._get_confidence_value(prediction.get("confidence"))

        candidates: List[Tuple[str, List[Union[int, float]], float]] = []
        for item in iterable_items:
            text = item.get("text")
            bbox = item.get("bbox")
            item_conf = self._get_confidence_value(
                item.get("confidence", default_conf)
            )

            if not isinstance(text, str) or not text.strip():
                continue

            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
            ):
                continue

            try:
                bbox_nums = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue

            candidates.append((text, bbox_nums, item_conf))

        return candidates

    def _map_bbox_to_page(
        self,
        bbox: List[Union[int, float]],
        ocr_rect: BoundingBox,
    ) -> Optional[BoundingBox]:
        try:
            left, top, right, bottom = bbox
        except ValueError:
            return None

        scaled_left = (float(left) / self.scale) + ocr_rect.l
        scaled_top = (float(top) / self.scale) + ocr_rect.t
        scaled_right = (float(right) / self.scale) + ocr_rect.l
        scaled_bottom = (float(bottom) / self.scale) + ocr_rect.t

        if scaled_right <= scaled_left or scaled_bottom <= scaled_top:
            return None

        return BoundingBox.from_tuple(
            coord=(
                scaled_left,
                scaled_top,
                scaled_right,
                scaled_bottom,
            ),
            origin=CoordOrigin.TOPLEFT,
        )

    @staticmethod
    def _get_confidence_value(value: object) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def __del__(self):
        if self._session is not None:
            self._session.close()
