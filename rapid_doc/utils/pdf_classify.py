# Copyright (c) Opendatalab. All rights reserved.
import re
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from pdfminer.converter import PDFPageAggregator
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams, LTFigure, LTImage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from rapid_doc.utils.pdfium_guard import (
    close_pdfium_document,
    open_pdfium_document,
    pdfium_guard,
)

MAX_SAMPLE_PAGES = 10
CHARS_THRESHOLD = 50
HIGH_IMAGE_COVERAGE_THRESHOLD = 0.8
CID_RATIO_THRESHOLD = 0.05


def classify(pdf_bytes):
    """
    Classify a PDF as text-based or OCR-based.

    Returns:
        "txt" if the PDF can be parsed as text, otherwise "ocr".
    """
    return classify_legacy(pdf_bytes)


def classify_legacy(pdf_bytes):
    """
    Legacy classification path kept for rollback and A/B comparison.
    """

    sample_pdf_bytes = extract_pages(pdf_bytes)
    if not sample_pdf_bytes:
        return "ocr"
    pdf = None
    try:
        pdf = open_pdfium_document(pdfium.PdfDocument, sample_pdf_bytes)
        with pdfium_guard():
            page_count = len(pdf)
            if page_count == 0:
                return "ocr"

            pages_to_check = min(page_count, MAX_SAMPLE_PAGES)

            if get_avg_cleaned_chars_per_page(pdf, pages_to_check) < CHARS_THRESHOLD:
                return "ocr"

        if detect_invalid_chars(sample_pdf_bytes):
            return "ocr"

        if (
            get_high_image_coverage_ratio(sample_pdf_bytes, pages_to_check)
            >= HIGH_IMAGE_COVERAGE_THRESHOLD
        ):
            return "ocr"

        return "txt"

    except Exception as e:
        logger.warning(f"Failed to classify PDF with legacy strategy: {e}")
        return "ocr"

    finally:
        close_pdfium_document(pdf)


def get_avg_cleaned_chars_per_page(pdf_doc, pages_to_check):
    total_chars = 0
    cleaned_total_chars = 0

    for i in range(pages_to_check):
        page = pdf_doc[i]
        text_page = page.get_textpage()
        try:
            text = text_page.get_text_bounded()
            total_chars += len(text)
            cleaned_text = re.sub(r"\s+", "", text)
            cleaned_total_chars += len(cleaned_text)
        finally:
            if hasattr(text_page, "close"):
                text_page.close()

    avg_cleaned_chars_per_page = cleaned_total_chars / pages_to_check
    return avg_cleaned_chars_per_page


def get_high_image_coverage_ratio(sample_pdf_bytes, pages_to_check):
    pdf_stream = BytesIO(sample_pdf_bytes)
    device = None
    try:
        parser = PDFParser(pdf_stream)
        document = PDFDocument(parser)

        if not document.is_extractable:
            return 1.0

        rsrcmgr = PDFResourceManager()
        laparams = LAParams(
            line_overlap=0.5,
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            boxes_flow=None,
            detect_vertical=False,
            all_texts=False,
        )
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        high_image_coverage_pages = 0
        page_count = 0

        for page in PDFPage.create_pages(document):
            if page_count >= pages_to_check:
                break

            interpreter.process_page(page)
            layout = device.get_result()

            page_width = layout.width
            page_height = layout.height
            page_area = page_width * page_height

            image_area = 0
            for element in layout:
                if isinstance(element, (LTImage, LTFigure)):
                    img_width = element.width
                    img_height = element.height
                    image_area += img_width * img_height

            coverage_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0
            if coverage_ratio >= HIGH_IMAGE_COVERAGE_THRESHOLD:
                high_image_coverage_pages += 1

            page_count += 1

        if page_count == 0:
            return 0.0

        return high_image_coverage_pages / page_count
    finally:
        if device is not None:
            device.close()
        pdf_stream.close()


def extract_pages(src_pdf_bytes: bytes) -> bytes:
    """
    Extract up to 10 random pages and return them as a new PDF.
    """

    pdf = None
    sample_docs = None
    try:
        with pdfium_guard():
            pdf = open_pdfium_document(pdfium.PdfDocument, src_pdf_bytes)
            total_page = len(pdf)
            if total_page == 0:
                logger.warning("PDF is empty, return empty document")
                return b""

            if total_page <= MAX_SAMPLE_PAGES:
                return src_pdf_bytes

            select_page_cnt = min(MAX_SAMPLE_PAGES, total_page)
            page_indices = np.random.choice(
                total_page, select_page_cnt, replace=False
            ).tolist()

            sample_docs = open_pdfium_document(pdfium.PdfDocument.new)
            sample_docs.import_pages(pdf, page_indices)

            output_buffer = BytesIO()
            sample_docs.save(output_buffer)
            return output_buffer.getvalue()
    except Exception as e:
        logger.exception(e)
        return src_pdf_bytes
    finally:
        close_pdfium_document(pdf)
        close_pdfium_document(sample_docs)


def detect_invalid_chars(sample_pdf_bytes: bytes) -> bool:
    """
    Detect whether a PDF contains invalid CID-style extracted text.
    """

    sample_pdf_file_like_object = BytesIO(sample_pdf_bytes)
    try:
        laparams = LAParams(
            line_overlap=0.5,
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            boxes_flow=None,
            detect_vertical=False,
            all_texts=False,
        )
        text = extract_text(pdf_file=sample_pdf_file_like_object, laparams=laparams)
        text = text.replace("\n", "")

        cid_pattern = re.compile(r"\(cid:\d+\)")
        matches = cid_pattern.findall(text)
        cid_count = len(matches)
        cid_len = sum(len(match) for match in matches)
        text_len = len(text)
        if text_len == 0:
            cid_chars_ratio = 0
        else:
            cid_chars_ratio = cid_count / (cid_count + text_len - cid_len)

        return cid_chars_ratio > CID_RATIO_THRESHOLD
    finally:
        sample_pdf_file_like_object.close()


if __name__ == "__main__":
    with open("/Users/myhloli/pdf/luanma2x10.pdf", "rb") as f:
        p_bytes = f.read()
        logger.info(f"PDF classify result: {classify(p_bytes)}")
