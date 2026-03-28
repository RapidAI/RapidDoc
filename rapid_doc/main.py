import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rapid_doc.backend.office.office_analyze import office_analyze
from rapid_doc.backend.office.office_middle_json_mkcontent import (
    union_make as office_union_make,
)
from rapid_doc.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from rapid_doc.backend.pipeline.pipeline_analyze import ModelSingleton
from rapid_doc.backend.pipeline.pipeline_analyze import (
    doc_analyze as pipeline_doc_analyze,
)
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from rapid_doc.cli.common import image_suffixes, office_suffixes, old_office_suffixes, read_fn
from rapid_doc.data.data_reader_writer.base import DataWriter
from rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes
from rapid_doc.utils.office_converter import convert_legacy_office_to_modern
from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes


@dataclass
class RapidDocOutput:
    # 统一的解析输出对象，单文档调用时直接返回它；
    # 同时实现了可迭代协议，因此也支持：
    # markdown, images = engine(pdf_bytes)
    middle_json: dict[str, Any] | None = None
    content_list_json: list[Any] | None = None
    markdown: str = ""
    images: dict[str, bytes] = field(default_factory=dict)
    draw_layout_bbox: bytes | None = None
    draw_span_bbox: bytes | None = None

    def __iter__(self):
        yield self.markdown
        yield self.images


class MemoryDataWriter(DataWriter):
    # 将写入的数据保存在内存中，便于直接返回给 Python 调用方。
    def __init__(self, parent_dir: str = "images") -> None:
        self._parent_dir = parent_dir
        self.data: dict[str, bytes] = {}

    def write(self, path: str, data: bytes) -> None:
        self.data[path.replace("\\", "/")] = data


class FanoutDataWriter(DataWriter):
    # 同时写入多个 writer。
    def __init__(self, *writers: DataWriter | None) -> None:
        self._writers = [writer for writer in writers if writer is not None]
        self._parent_dir = ""
        for writer in self._writers:
            parent_dir = getattr(writer, "_parent_dir", "")
            if parent_dir:
                self._parent_dir = parent_dir
                break

    def write(self, path: str, data: bytes) -> None:
        for writer in self._writers:
            writer.write(path, data)


class RapidDOC:
    def __init__(
        self,
        layout_config: dict[str, Any] | None = None,
        ocr_config: dict[str, Any] | None = None,
        formula_config: dict[str, Any] | None = None,
        table_config: dict[str, Any] | None = None,
        checkbox_config: dict[str, Any] | None = None,
        image_config: dict[str, Any] | None = None,
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        lang: str = "ch",
        make_md_mode: str = MakeMode.MM_MD,
        image_writer: DataWriter | None = None,
        md_writer: DataWriter | None = None,
        image_dir_name: str = "images",
        image_output_mode: str = "external",
        preload_model: bool = False,
    ) -> None:
        self.layout_config = layout_config or {}
        self.ocr_config = ocr_config or {}
        self.formula_config = formula_config or {}
        self.table_config = table_config or {}
        self.checkbox_config = checkbox_config or {}
        self.image_config = image_config or {}

        self.parse_method = parse_method
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.lang = lang
        self.make_md_mode = make_md_mode
        self.default_image_writer = image_writer
        self.default_md_writer = md_writer
        self.image_dir_name = image_dir_name or "images"
        self.image_output_mode = image_output_mode

        self._validate_image_output_mode(self.image_output_mode)

        # 在实例化阶段主动预热模型，后续 __call__ 时能复用缓存。
        if preload_model:
            self.warmup(lang=lang)

    def warmup(
        self,
        lang: str | None = None,
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
    ) -> None:
        warmup_lang = lang or self.lang
        warmup_formula_enable = (
            self.formula_enable if formula_enable is None else formula_enable
        )
        warmup_table_enable = (
            self.table_enable if table_enable is None else table_enable
        )

        ModelSingleton().get_model(
            lang=warmup_lang,
            formula_enable=warmup_formula_enable,
            table_enable=warmup_table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
        )

    def __call__(
        self,
        inputs: str | bytes | Path | list[str | bytes | Path],
        image_writer: DataWriter | None = None,
        md_writer: DataWriter | None = None,
        image_output_mode: str | None = None,
        image_dir_name: str | None = None,
        parse_method: str | None = None,
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
        lang: str | list[str] | None = None,
        start_page_id: int = 0,
        end_page_id: int | None = None,
        draw_layout_bbox: bool | None = None,
        draw_span_bbox: bool | None = None,
        draw_layout_bbox_enable: bool = False,
        draw_span_bbox_enable: bool = False,
    ) -> RapidDocOutput | list[RapidDocOutput]:
        is_batch = self._is_batch_input(inputs)
        normalized_inputs = list(inputs) if is_batch else [inputs]

        final_image_output_mode = image_output_mode or self.image_output_mode
        final_image_dir_name = image_dir_name or self.image_dir_name
        final_parse_method = parse_method or self.parse_method
        final_formula_enable = (
            self.formula_enable if formula_enable is None else formula_enable
        )
        final_table_enable = (
            self.table_enable if table_enable is None else table_enable
        )
        final_image_writer = image_writer or self.default_image_writer
        final_md_writer = md_writer or self.default_md_writer

        self._validate_image_output_mode(final_image_output_mode)

        if draw_layout_bbox is not None:
            draw_layout_bbox_enable = draw_layout_bbox
        if draw_span_bbox is not None:
            draw_span_bbox_enable = draw_span_bbox

        normalized_docs = self._normalize_inputs(normalized_inputs)
        lang_list = self._normalize_lang_list(lang, len(normalized_docs))

        outputs: list[RapidDocOutput] = []
        pipeline_indexes: list[int] = []
        office_indexes: list[int] = []
        pipeline_pdf_bytes_list: list[bytes | dict[str, Any]] = []
        pipeline_name_list: list[str] = []
        pipeline_lang_list: list[str] = []

        for index, doc in enumerate(normalized_docs):
            if doc["suffix"] in office_suffixes:
                office_indexes.append(index)
                outputs.append(self._empty_output())
                continue

            if doc["suffix"] not in ["pdf", *image_suffixes]:
                raise ValueError(f"Unsupported input suffix: {doc['suffix']}")

            pipeline_indexes.append(index)
            outputs.append(self._empty_output())
            pipeline_pdf_bytes_list.append(doc["pdf_bytes"])
            pipeline_name_list.append(doc["name"])
            pipeline_lang_list.append(lang_list[index])

        for index in office_indexes:
            doc = normalized_docs[index]
            outputs[index] = self._parse_office(
                name=doc["name"],
                file_bytes=doc["raw_bytes"],
                image_writer=final_image_writer,
                md_writer=final_md_writer,
                image_dir_name=final_image_dir_name,
                image_output_mode=final_image_output_mode,
            )

        if pipeline_pdf_bytes_list:
            pipeline_outputs = self._parse_pipeline_batch(
                names=pipeline_name_list,
                pdf_bytes_list=pipeline_pdf_bytes_list,
                lang_list=pipeline_lang_list,
                parse_method=final_parse_method,
                formula_enable=final_formula_enable,
                table_enable=final_table_enable,
                image_writer=final_image_writer,
                md_writer=final_md_writer,
                image_dir_name=final_image_dir_name,
                image_output_mode=final_image_output_mode,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                draw_layout_bbox_enable=draw_layout_bbox_enable,
                draw_span_bbox_enable=draw_span_bbox_enable,
            )
            for output_index, result in zip(pipeline_indexes, pipeline_outputs):
                outputs[output_index] = result

        if is_batch:
            return outputs
        return outputs[0]

    def _parse_pipeline_batch(
        self,
        names: list[str],
        pdf_bytes_list: list[bytes | dict[str, Any]],
        lang_list: list[str],
        parse_method: str,
        formula_enable: bool,
        table_enable: bool,
        image_writer: DataWriter | None,
        md_writer: DataWriter | None,
        image_dir_name: str,
        image_output_mode: str,
        start_page_id: int,
        end_page_id: int | None,
        draw_layout_bbox_enable: bool,
        draw_span_bbox_enable: bool,
    ) -> list[RapidDocOutput]:
        from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2

        sliced_pdf_bytes_list = [
            convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            for pdf_bytes in pdf_bytes_list
        ]

        (
            infer_results,
            all_image_lists,
            all_page_dicts,
            final_lang_list,
            ocr_enabled_list,
        ) = pipeline_doc_analyze(
            sliced_pdf_bytes_list,
            lang_list,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
            checkbox_config=self.checkbox_config,
        )

        outputs: list[RapidDocOutput] = []
        for index, model_list in enumerate(infer_results):
            name = names[index]
            pdf_bytes = sliced_pdf_bytes_list[index]
            pdf_raw_bytes = self._extract_pdf_bytes(pdf_bytes)
            memory_image_writer, combined_image_writer = self._build_image_writer(
                image_dir_name=image_dir_name,
                extra_image_writer=image_writer,
            )

            middle_json = pipeline_result_to_middle_json(
                model_list,
                all_image_lists[index],
                all_page_dicts[index],
                combined_image_writer,
                final_lang_list[index],
                ocr_enabled_list[index],
                formula_enable,
                ocr_config=self.ocr_config,
                image_config=self.image_config,
            )
            markdown = pipeline_union_make(
                middle_json["pdf_info"],
                self.make_md_mode,
                image_dir_name,
            )
            content_list_json = pipeline_union_make(
                middle_json["pdf_info"],
                MakeMode.CONTENT_LIST,
                image_dir_name,
            )

            image_bytes_map = dict(memory_image_writer.data)
            logical_image_map = self._build_logical_image_map(
                image_bytes_map,
                image_dir_name=image_dir_name,
            )

            if image_output_mode == "data_uri":
                markdown = self._replace_markdown_images_with_data_uri(
                    markdown,
                    logical_image_map,
                )

            layout_pdf_bytes = None
            span_pdf_bytes = None
            if draw_layout_bbox_enable:
                layout_pdf_bytes = self._render_draw_pdf_bytes(
                    drawer=draw_layout_bbox,
                    pdf_info=middle_json["pdf_info"],
                    pdf_bytes=pdf_raw_bytes,
                    filename=f"{name}_layout.pdf",
                )
            if draw_span_bbox_enable:
                span_pdf_bytes = self._render_draw_pdf_bytes(
                    drawer=draw_span_bbox,
                    pdf_info=middle_json["pdf_info"],
                    pdf_bytes=pdf_raw_bytes,
                    filename=f"{name}_span.pdf",
                )

            output = RapidDocOutput(
                middle_json=middle_json,
                content_list_json=content_list_json,
                markdown=markdown,
                images=image_bytes_map,
                draw_layout_bbox=layout_pdf_bytes,
                draw_span_bbox=span_pdf_bytes,
            )

            self._dump_output_if_needed(
                output=output,
                name=name,
                md_writer=md_writer,
            )
            outputs.append(output)

        return outputs

    def _parse_office(
        self,
        name: str,
        file_bytes: bytes,
        image_writer: DataWriter | None,
        md_writer: DataWriter | None,
        image_dir_name: str,
        image_output_mode: str,
    ) -> RapidDocOutput:
        memory_image_writer, combined_image_writer = self._build_image_writer(
            image_dir_name=image_dir_name,
            extra_image_writer=image_writer,
        )

        middle_json, model_json = office_analyze(
            file_bytes,
            image_writer=combined_image_writer,
        )
        markdown = office_union_make(
            middle_json["pdf_info"],
            self.make_md_mode,
            image_dir_name,
        )
        content_list_json = office_union_make(
            middle_json["pdf_info"],
            MakeMode.CONTENT_LIST,
            image_dir_name,
        )

        image_bytes_map = dict(memory_image_writer.data)
        logical_image_map = self._build_logical_image_map(
            image_bytes_map,
            image_dir_name=image_dir_name,
        )

        if image_output_mode == "data_uri":
            markdown = self._replace_markdown_images_with_data_uri(
                markdown,
                logical_image_map,
            )

        output = RapidDocOutput(
            middle_json=middle_json,
            content_list_json=content_list_json,
            markdown=markdown,
            images=image_bytes_map,
            draw_layout_bbox=None,
            draw_span_bbox=None,
        )

        self._dump_output_if_needed(
            output=output,
            name=name,
            md_writer=md_writer,
        )
        return output

    def _dump_output_if_needed(
        self,
        output: RapidDocOutput,
        name: str,
        md_writer: DataWriter | None,
    ) -> None:
        if md_writer is None:
            return

        md_writer.write_string(f"{name}.md", output.markdown)
        md_writer.write_string(
            f"{name}_middle.json",
            json.dumps(output.middle_json, ensure_ascii=False, indent=4),
        )
        md_writer.write_string(
            f"{name}_content_list.json",
            json.dumps(output.content_list_json, ensure_ascii=False, indent=4),
        )
        if output.draw_layout_bbox is not None:
            md_writer.write(f"{name}_layout.pdf", output.draw_layout_bbox)
        if output.draw_span_bbox is not None:
            md_writer.write(f"{name}_span.pdf", output.draw_span_bbox)

    def _build_image_writer(
        self,
        image_dir_name: str,
        extra_image_writer: DataWriter | None,
    ) -> tuple[MemoryDataWriter, DataWriter]:
        memory_image_writer = MemoryDataWriter(parent_dir=image_dir_name)
        return memory_image_writer, FanoutDataWriter(memory_image_writer, extra_image_writer)

    def _normalize_inputs(
        self,
        inputs: list[str | bytes | Path],
    ) -> list[dict[str, Any]]:
        normalized_docs: list[dict[str, Any]] = []
        for index, item in enumerate(inputs):
            doc = self._normalize_single_input(item, index)
            normalized_docs.append(doc)
        return normalized_docs

    def _normalize_single_input(
        self,
        item: str | bytes | Path,
        index: int,
    ) -> dict[str, Any]:
        if isinstance(item, (str, Path)):
            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

            file_suffix = path.suffix.lower().lstrip(".")
            actual_path = path
            if file_suffix in old_office_suffixes:
                actual_path = Path(convert_legacy_office_to_modern(path))
                file_suffix = actual_path.suffix.lower().lstrip(".")

            raw_bytes = actual_path.read_bytes()
            if file_suffix in image_suffixes:
                pdf_bytes = read_fn(actual_path)
            else:
                pdf_bytes = raw_bytes

            return {
                "name": actual_path.stem,
                "suffix": file_suffix or guess_suffix_by_bytes(raw_bytes),
                "raw_bytes": raw_bytes,
                "pdf_bytes": pdf_bytes,
            }

        if isinstance(item, bytearray):
            item = bytes(item)

        if isinstance(item, bytes):
            suffix = guess_suffix_by_bytes(item)
            doc_name = f"document_{index + 1}"
            if suffix in image_suffixes:
                return {
                    "name": doc_name,
                    "suffix": suffix,
                    "raw_bytes": item,
                    "pdf_bytes": images_bytes_to_pdf_bytes(item),
                }
            if suffix in ["pdf", *office_suffixes]:
                return {
                    "name": doc_name,
                    "suffix": suffix,
                    "raw_bytes": item,
                    "pdf_bytes": item,
                }
            raise ValueError(f"Unsupported bytes input suffix: {suffix}")

        raise TypeError(f"Unsupported input type: {type(item)}")

    def _normalize_lang_list(
        self,
        lang: str | list[str] | None,
        doc_count: int,
    ) -> list[str]:
        if lang is None:
            return [self.lang] * doc_count
        if isinstance(lang, str):
            return [lang] * doc_count
        if len(lang) != doc_count:
            raise ValueError("The length of lang list must match the number of inputs.")
        return list(lang)

    def _replace_markdown_images_with_data_uri(
        self,
        markdown: str,
        logical_image_map: dict[str, bytes],
    ) -> str:
        updated_markdown = markdown
        replacements = sorted(logical_image_map.items(), key=lambda item: len(item[0]), reverse=True)
        for logical_ref, image_bytes in replacements:
            data_uri = self._to_data_uri(logical_ref, image_bytes)
            updated_markdown = updated_markdown.replace(logical_ref, data_uri)
        return updated_markdown

    def _build_logical_image_map(
        self,
        image_bytes_map: dict[str, bytes],
        image_dir_name: str,
    ) -> dict[str, bytes]:
        logical_image_map: dict[str, bytes] = {}
        for image_name, image_bytes in image_bytes_map.items():
            logical_ref = f"{image_dir_name}/{image_name}".replace("\\", "/")
            logical_image_map[logical_ref] = image_bytes
            logical_image_map[image_name] = image_bytes
        return logical_image_map

    def _render_draw_pdf_bytes(
        self,
        drawer,
        pdf_info: list[dict[str, Any]],
        pdf_bytes: bytes,
        filename: str,
    ) -> bytes:
        with tempfile.TemporaryDirectory(prefix="rapid_doc_draw_") as temp_dir:
            drawer(pdf_info, pdf_bytes, temp_dir, filename)
            return Path(temp_dir, filename).read_bytes()

    def _extract_pdf_bytes(self, pdf_bytes: bytes | dict[str, Any]) -> bytes:
        if isinstance(pdf_bytes, dict):
            return pdf_bytes["pdf_bytes"]
        return pdf_bytes

    def _to_data_uri(self, logical_ref: str, image_bytes: bytes) -> str:
        suffix = Path(logical_ref).suffix.lower().lstrip(".") or "png"
        mime_suffix = "jpeg" if suffix == "jpg" else suffix
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{mime_suffix};base64,{encoded}"

    def _empty_output(self) -> RapidDocOutput:
        return RapidDocOutput()

    def _validate_image_output_mode(self, image_output_mode: str) -> None:
        if image_output_mode not in {"external", "data_uri"}:
            raise ValueError(
                "image_output_mode only supports 'external' and 'data_uri'."
            )

    def _is_batch_input(self, inputs: Any) -> bool:
        if isinstance(inputs, (bytes, bytearray, str, Path)):
            return False
        return isinstance(inputs, Iterable)


if __name__ == '__main__':

    __dir__ = Path(__file__).resolve().parent.parent
    output_dir = os.path.join(__dir__, "output")

    doc_path_list = [
        __dir__ / "demo/pdfs/示例1-论文模板.pdf",
        __dir__ / "demo/docx/test.docx",
    ]
    engine = RapidDOC()
    outputs = engine(doc_path_list)
    for output in outputs:
        print(output.markdown)