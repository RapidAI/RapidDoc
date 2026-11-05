"""RapidDoc Python API backed by native PDF and six Office format parsers."""

import base64
import json
import mimetypes
import os
import time
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from functools import lru_cache
from urllib.parse import urlparse, unquote
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _office_delimiters():
    path = Path(os.getenv("MINERU_TOOLS_CONFIG_JSON", "magic.json"))
    if not path.is_absolute():
        path = Path.home() / path
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    value = config.get("latex-delimiter-config")
    if not value:
        return None
    return json.dumps(
        {
            f"{group}_{side}": value[group][side]
            for group in ("inline", "display")
            for side in ("left", "right")
        }
    )


@dataclass
class RapidDocOutput:
    markdown: str = ""
    images: dict[str, bytes] = field(default_factory=dict)
    middle_json: dict[str, Any] | None = None
    content_list_json: list[Any] | None = None

    def __iter__(self):
        yield self.markdown
        yield self.images


def _scalar(value):
    return getattr(value, "value", value)


def _doc_orientation_enabled():
    return os.getenv("USE_DOC_ORIENTATION_CLASSIFY", "false").strip().lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


def _read_input(item, index):
    name = f"document_{index + 1}"
    office_hint = False
    suffix = None
    if isinstance(item, bytearray):
        item = bytes(item)
    if isinstance(item, (str, Path)):
        text = str(item)
        parsed = urlparse(text)
        suffix = Path(unquote(parsed.path)).suffix.lower()
        office_hint = suffix in (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx")
        if parsed.scheme in ("http", "https"):
            from ._native import read_document_url
            logger.info("Downloading document...")
            filename, data = read_document_url(text)
            name = Path(filename).stem or name
            suffix = Path(filename).suffix.lower()
            office_hint = suffix in (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx")
            logger.info("Downloaded document: {} ({} bytes)", filename, len(data))
        else:
            path = Path(item)
            name = path.stem
            data = path.read_bytes()
    elif isinstance(item, bytes):
        data = item
    elif callable(getattr(item, "read", None)):
        try:
            item.seek(0)
        except (AttributeError, OSError, ValueError):
            pass
        data = item.read()
        hint = getattr(item, "name", None)
        if isinstance(hint, (str, Path)):
            name = Path(hint).stem or name
            suffix = Path(hint).suffix.lower()
            office_hint = suffix in (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx")
    else:
        raise TypeError(f"Unsupported input type: {type(item)}")
    if not isinstance(data, bytes):
        raise TypeError("Input stream must return bytes")
    if data.startswith((b"PK\x03\x04", b"PK\x05\x06", b"\xd0\xcf\x11\xe0")) or (
        not data and office_hint
    ):
        return name, data, suffix[1:] if not data and office_hint else None
    if data.lstrip().startswith(b"%PDF-"):
        return name, data, None
    from PIL import Image, ImageOps

    try:
        with Image.open(BytesIO(data)) as image:
            frames = []
            for i in range(getattr(image, "n_frames", 1)):
                image.seek(i)
                frames.append(ImageOps.exif_transpose(image).convert("RGB"))
            buffer = BytesIO()
            try:
                frames[0].save(
                    buffer, format="PDF", resolution=200.0, save_all=True, append_images=frames[1:]
                )
                return name, buffer.getvalue(), None
            finally:
                for frame in frames:
                    frame.close()
    except (OSError, ValueError) as e:
        raise ValueError(
            "Unsupported or invalid document; supported formats: PDF, images, DOC, DOCX, XLS, XLSX, PPT, PPTX"
        ) from e


class RapidDoc:
    def __init__(
        self,
        layout_config=None,
        ocr_config=None,
        formula_config=None,
        table_config=None,
        checkbox_config=None,
        image_config=None,
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        lang="ch",
        make_md_mode="mm_markdown",
        output_dir=None,
        image_writer=None,
        md_writer=None,
        image_dir_name="images",
        image_output_mode="url",
        preload_model=False,
        pdf_pages_batch=64,
        cpu_spinning=None,
        cpu_dynamic_block_base=None,
    ):
        self.layout_config = dict(layout_config or {})
        self.ocr_config = dict(ocr_config or {})
        self.formula_config = dict(formula_config or {})
        self.table_config = dict(table_config or {})
        self.checkbox_config = dict(checkbox_config or {})
        self.image_config = dict(image_config or {})
        self.parse_method = parse_method
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.lang = lang
        self.make_md_mode = _scalar(make_md_mode)
        self.default_output_dir = output_dir
        self.default_image_writer = image_writer
        self.default_md_writer = md_writer
        self.image_dir_name = image_dir_name or "images"
        self.image_output_mode = image_output_mode
        self.pdf_pages_batch = pdf_pages_batch
        if cpu_spinning is not None and type(cpu_spinning) is not bool:
            raise ValueError("cpu_spinning must be bool or None")
        if cpu_dynamic_block_base is not None and (
            type(cpu_dynamic_block_base) is not int or not 0 <= cpu_dynamic_block_base <= 2147483647
        ):
            raise ValueError("cpu_dynamic_block_base must be a nonnegative int32 or None")
        self.cpu_spinning = cpu_spinning
        self.cpu_dynamic_block_base = cpu_dynamic_block_base
        self._models = None
        self._native_engine = None
        self._native_config = None
        self._closed = False
        self._state_lock = threading.RLock()
        self._validate(parse_method, formula_enable, table_enable, image_output_mode)
        if type(pdf_pages_batch) is not int or pdf_pages_batch < 1:
            raise ValueError("pdf_pages_batch must be positive")
        formula_batch = self.formula_config.get("batch_num", 1)
        if type(formula_batch) is not int or formula_batch < 1:
            raise ValueError("formula batch_num must be a positive integer")
        if self.checkbox_config.get("enable"):
            raise NotImplementedError("复选框不在当前集成范围内")
        if set(self.image_config) - {"extract_original_image", "extract_original_image_iou_thresh"}:
            raise ValueError("Unknown image_config option")
        if self.ocr_config.get("custom_model"):
            raise NotImplementedError("自定义 OCR 协议尚未接入兼容入口")
        if self.table_config.get("enable_blank_cell_rec"):
            raise NotImplementedError("空格补识别尚未在兼容入口验收")
        self._mode = self.layout_config.get("docvortex_mode")
        native_only = _scalar(self.layout_config.get("model_type")) == "docvortex_rule"
        if self._mode is None:
            self._mode = "rule" if native_only else "off"
        if self._mode not in ("off", "rule", "rule_plus_s"):
            raise ValueError("Invalid docvortex_mode")
        if native_only and self._mode == "off":
            raise ValueError("model_type=docvortex_rule conflicts with docvortex_mode=off")
        self._native_only = native_only
        if self._mode == "rule_plus_s":
            raise NotImplementedError("实际 S 模型补充/印章宿主验收将在后续指定步骤完成")
        if preload_model:
            self.warmup(lang=lang)

    @property
    def closed(self):
        return self._closed

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("RapidDoc is closed")

    def close(self):
        """Release sessions promptly. Repeated calls are safe."""
        with self._state_lock:
            if self._closed:
                return
            if self._native_engine is not None:
                self._native_engine.close()
            self._closed = True
            self._native_engine = None
            self._native_config = None
            self._models = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @staticmethod
    def _validate(method, formula, table, mode):
        if method not in ("auto", "txt", "ocr"):
            raise ValueError("parse_method must be auto/txt/ocr")
        if type(formula) is not bool or type(table) is not bool:
            raise ValueError("formula_enable/table_enable must be bool")
        if mode not in ("url", "data_uri"):
            raise ValueError("image_output_mode only supports 'url' and 'data_uri'.")

    def _get_models(self):
        self._ensure_open()
        from ._runtime import load_models

        if self._models is None:
            self._models = load_models(
                self.layout_config,
                self.ocr_config,
                self.formula_config,
                self.table_config,
                cpu_spinning=self.cpu_spinning,
                cpu_dynamic_block_base=self.cpu_dynamic_block_base,
            )
        return self._models

    def _engine_config(self):
        from ._runtime import model_config, reference_crop_sampling

        return model_config(
            self.layout_config,
            self.ocr_config,
            self.formula_config,
            self.table_config,
            image_config=self.image_config,
            parse_method=self.parse_method,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            lang=self.lang,
            make_md_mode=self.make_md_mode,
            pdf_pages_batch=self.pdf_pages_batch,
            cpu_spinning=self.cpu_spinning,
            cpu_dynamic_block_base=self.cpu_dynamic_block_base,
            doc_orientation_classify=_doc_orientation_enabled(),
            crop_sampling=reference_crop_sampling(),
        )

    def _get_native_engine(self):
        with self._state_lock:
            self._ensure_open()
            config = self._engine_config()
            if self._native_engine is None:
                from ._native import DocumentEngine
                from ._runtime import resource_hints

                self._native_engine = DocumentEngine(config, resource_hints())
            elif config != self._native_config:
                self._native_engine.set_config(config)
            self._native_config = config
            return self._native_engine

    def warmup(self, lang=None, formula_enable=None, table_enable=None):
        from ._runtime import log_model_event, prepare_device

        options = dict(
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            doc_orientation_classify=_doc_orientation_enabled(),
        )
        self._get_native_engine().warmup(json.dumps(options), log_model_event, prepare_device)

    def __call__(
        self,
        inputs,
        output_dir=None,
        image_writer=None,
        md_writer=None,
        image_output_mode=None,
        image_dir_name=None,
        parse_method=None,
        formula_enable=None,
        table_enable=None,
        lang=None,
        start_page_id=0,
        end_page_id=None,
        f_dump_middle_json=True,
        f_dump_content_list=True,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
    ):
        from ._native import ParseProgress
        from ._runtime import resource_hints, prepare_device, log_model_event

        self._ensure_open()
        method = parse_method or self.parse_method
        formula = self.formula_enable if formula_enable is None else formula_enable
        table = self.table_enable if table_enable is None else table_enable
        mode = image_output_mode or self.image_output_mode
        prefix = image_dir_name or self.image_dir_name
        outdir = output_dir or self.default_output_dir
        iw = image_writer or self.default_image_writer
        mw = md_writer or self.default_md_writer
        self._validate(method, formula, table, mode)
        if Path(prefix).name != prefix or prefix in (".", ".."):
            raise ValueError("image_dir_name must be a directory name")
        if type(start_page_id) is not int or start_page_id < 0:
            raise ValueError("Invalid start_page_id")
        if end_page_id is not None and (
            type(end_page_id) is not int or end_page_id < start_page_id
        ):
            raise ValueError("Invalid end_page_id")
        batch = not isinstance(inputs, (str, bytes, bytearray, Path)) and not callable(
            getattr(inputs, "read", None)
        )
        items = list(inputs) if batch else [inputs]
        languages = (
            [lang or self.lang] * len(items)
            if lang is None or isinstance(lang, str)
            else list(lang)
        )
        if len(languages) != len(items):
            raise ValueError("The length of lang list must match the number of inputs.")
        results = []
        for index, (item, language) in enumerate(zip(items, languages)):
            started = time.perf_counter()
            name, data, empty_format = _read_input(item, index)
            stage = "classify"
            current = None
            progress_view = None
            try:
                logger.info("{}: parsing ({})", name, method)
                office = not data or data.startswith(
                    (b"PK\x03\x04", b"PK\x05\x06", b"\xd0\xcf\x11\xe0")
                )
                count = 0
                if not office:
                    progress_view = ParseProgress(name, 0)
                processing_times = {}
                batch_times = {}

                def collect_times():
                    for key, seconds in batch_times.items():
                        processing_times[key] = processing_times.get(key, 0.0) + seconds
                    batch_times.clear()

                def progress(event):
                    nonlocal stage, current, count, progress_view
                    if event["stage"].startswith("download_"):
                        detail = json.loads(event["detail"])
                        if progress_view is not None:
                            progress_view.clear()
                        download_stage = event["stage"]
                        if download_stage == "download_started":
                            logger.info("Initiating download: {}", detail["url"])
                            logger.info("Model destination: {}", detail["path"])
                        elif download_stage == "download_progress":
                            total = detail.get("total")
                            size = (
                                f" / {total / 1048576:.2f} MiB"
                                if total is not None
                                else " (total size unknown)"
                            )
                            logger.info(
                                "Downloading {}: {:.2f} MiB{} [{:.1f}s]",
                                detail["model_id"],
                                detail["completed"] / 1048576,
                                size,
                                detail["elapsed_seconds"],
                            )
                        elif download_stage == "download_verifying":
                            logger.info("Verifying SHA-256: {}", detail["model_id"])
                        elif download_stage == "download_completed":
                            logger.info("Successfully saved to: {}", detail["path"])
                        elif download_stage == "download_failed":
                            logger.error(
                                "Download failed: {}: {}", detail["model_id"], detail["error"]
                            )
                        return
                    if event["stage"] == "model_resolved":
                        if progress_view is not None:
                            progress_view.clear()
                        log_model_event(event)
                        return
                    if event["stage"] == "document_opened":
                        count = event["total"]
                        progress_view = ParseProgress(name, count)
                        if start_page_id >= count and count:
                            raise ValueError("start_page_id exceeds source page count")
                        last = count if end_page_id is None else min(count, end_page_id + 1)
                        logger.debug(
                            "{}: method {}, pages {}-{} of {}",
                            name,
                            method,
                            start_page_id + 1,
                            last,
                            count,
                        )
                        return
                    detail = event.get("detail") or ""
                    if event["stage"] == "classification_fallback":
                        logger.warning(
                            "{}: PDF classification selected OCR after an extraction failure: {}",
                            name,
                            detail,
                        )
                        return
                    if event["stage"] == "summary":
                        if detail.startswith("model_initialization:"):
                            progress_view.clear()
                            logger.info(
                                "{}: model initialization {}", name, detail.rsplit(" in ", 1)[-1]
                            )
                            return
                        if os.getenv("RAPIDDOC_DETAILED_LOGS", "").strip().lower() in (
                            "1",
                            "true",
                            "yes",
                            "on",
                        ):
                            progress_view.clear()
                            logger.debug("{}: {}", name, detail)
                        return
                    if event["stage"] == "batch":
                        collect_times()
                        progress_view.reset()
                        logger.info("{}: {}", name, detail)
                        return
                    if event["stage"] == "native_analysis":
                        progress_view.clear()
                        logger.info("{}: Native PDF analysis ({})", name, detail)
                        return
                    if event["stage"] == "page_prepared":
                        if os.getenv("RAPIDDOC_DETAILED_LOGS", "").strip().lower() in (
                            "1",
                            "true",
                            "yes",
                            "on",
                        ):
                            progress_view.clear()
                            logger.debug(
                                "{}: page {}/{} prepared ({})",
                                name,
                                event["page_index"] + 1,
                                count,
                                detail,
                            )
                        return
                    stage = event["stage"]
                    if event.get("completed") is not None:
                        if event.get("page_index") is not None:
                            current = event["page_index"]
                        seconds = event.get("elapsed_seconds")
                        if seconds is not None:
                            batch_times[stage] = max(batch_times.get(stage, 0.0), seconds)
                        # tqdm elapsed/ETA/rate describe actual waiting time.
                        # Rust's measured work time is reported separately below.
                        progress_view.advance(stage, event["completed"], event["total"])
                    else:
                        current = event.get("page_index")
                        if stage in ("finalize", "write"):
                            progress_view.clear()
                        else:
                            stage = progress_view.update(stage, current, detail)
                    if stage == "layout_fallback":
                        progress_view.clear()
                        logger.warning(
                            "{}: native analysis failed; using layout model: {}",
                            name,
                            event.get("detail", ""),
                        )

                engine = self._get_native_engine()
                if not office:
                    engine.set_resources(resource_hints(pdfium=True))
                delimiters = _office_delimiters() if office else None
                options = dict(
                    start_page=start_page_id,
                    end_page=None if end_page_id is None else end_page_id + 1,
                    parse_method=method,
                    formula_enable=formula,
                    table_enable=table,
                    lang=language,
                    doc_orientation_classify=_doc_orientation_enabled(),
                    image_dir_name=prefix,
                    empty_format=empty_format,
                    latex_delimiters=json.loads(delimiters) if delimiters else None,
                    draw_layout=bool(f_draw_layout_bbox and (outdir or mw)),
                    draw_span=bool(f_draw_span_bbox and (outdir or mw)),
                )
                stage = "office" if office else "parse"
                metadata, blobs, drawings = engine.parse(
                    data, json.dumps(options, allow_nan=False), progress, prepare_device
                )
                collect_times()
                if processing_times:
                    progress_view.clear()
                    logger.info(
                        "{}: processing time (excludes interleaved stages and model loading): {}",
                        name,
                        ", ".join(
                            f"{key}={seconds:.3f}s" for key, seconds in processing_times.items()
                        ),
                    )
                parsed = json.loads(metadata)
                for message in parsed["diagnostics"]:
                    logger.warning("{}: {}", name, message)
                for route in parsed["metadata"].get("native_table_routes", []):
                    quality = route.get("text_retention") or {}
                    for key in ("main_rejected", "fallback_rejected"):
                        if quality.get(key):
                            logger.warning(
                                "{}: table candidate HTML rejected: {}; selected={}",
                                name,
                                quality[key],
                                quality.get("selected"),
                            )
                middle = parsed["middle_json"]
                pages = middle["pdf_info"]
                markdown = parsed["markdown"]
                content = parsed["content_list_json"]
                images = dict(blobs)
                if mode == "data_uri":
                    for key in sorted(images, key=len, reverse=True):
                        mime = mimetypes.guess_type(key)[0] or "image/png"
                        markdown = markdown.replace(
                            prefix + "/" + key,
                            "data:"
                            + mime
                            + ";base64,"
                            + base64.b64encode(images[key]).decode("ascii"),
                        )
                output = RapidDocOutput(markdown, images, middle, content)
                stage = "write"
                if progress_view is not None:
                    progress_view.clear()
                destination_method = "office" if office else method
                self._write(
                    output,
                    name,
                    destination_method,
                    outdir,
                    iw,
                    mw,
                    prefix,
                    f_dump_middle_json,
                    f_dump_content_list,
                )
                for kind, blob in drawings:
                    filename = name + "_" + kind + ".pdf"
                    if outdir:
                        (Path(outdir) / name / destination_method / filename).write_bytes(blob)
                    else:
                        mw.write(filename, blob)
                if progress_view is not None:
                    progress_view.clear()
                logger.info(
                    "{}: completed {} pages, {} Markdown characters, {} assets in {:.2f}s",
                    name,
                    len(pages),
                    len(markdown),
                    len(images),
                    time.perf_counter() - started,
                )
                if outdir:
                    logger.info("local output dir is {}", Path(outdir) / name / destination_method)
                results.append(output)
            except Exception as e:
                if progress_view is not None:
                    progress_view.clear()
                logger.error(
                    "{}: {} failed (page={}): {}",
                    name,
                    stage,
                    None if current is None else current + 1,
                    e,
                )
                if hasattr(e, "add_note"):
                    e.add_note(
                        f"RapidDoc source={name}, stage={stage}, page={None if current is None else current + 1}"
                    )
                raise
        return results if batch else results[0]

    @staticmethod
    def _write(
        output, name, method, outdir, image_writer, md_writer, prefix, dump_middle, dump_content
    ):
        folder = Path(outdir) / name / method if outdir else None
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / prefix).mkdir(exist_ok=True)
        for key, blob in output.images.items():
            if Path(key).name != key:
                raise ValueError("Invalid output image name")
            if folder:
                (folder / prefix / key).write_bytes(blob)
            elif image_writer:
                image_writer.write(key, blob)
        texts = {name + ".md": output.markdown}
        if dump_middle:
            texts[name + "_middle.json"] = json.dumps(
                output.middle_json, ensure_ascii=False, indent=4
            )
        if dump_content:
            texts[name + "_content_list.json"] = json.dumps(
                output.content_list_json, ensure_ascii=False, indent=4
            )
        for key, text in texts.items():
            if folder:
                (folder / key).write_text(text, encoding="utf-8")
            elif md_writer:
                if callable(getattr(md_writer, "write_string", None)):
                    md_writer.write_string(key, text)
                else:
                    md_writer.write(key, text.encode("utf-8"))
