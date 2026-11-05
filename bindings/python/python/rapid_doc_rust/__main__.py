"""安装包命令行入口：所有解析计算调用原生扩展。"""

import argparse
from dataclasses import asdict
import json
import sys
from . import (
    RapidDocError,
    OnnxSession,
    capabilities,
    parse_pdf_text,
    recover_pdf_table,
    ocr_png,
    ocr_pdf_page,
    recognize_table_png,
    recognize_wired_table_png,
)


def main():
    """业务输出写 stdout，稳定错误码写 stderr，失败退出码为 2。"""
    parser = argparse.ArgumentParser(prog="rapid-doc-rust")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities")
    text = commands.add_parser("extract-text")
    text.add_argument("input")
    text.add_argument("--start", type=int, default=0)
    text.add_argument("--end", type=int)
    text.add_argument("--allow-partial", action="store_true")
    text.add_argument("--json", action="store_true")
    text.add_argument("--pdfium")
    table = commands.add_parser("recover-table")
    table.add_argument("input")
    table.add_argument("--bounds", nargs=4, type=float, required=True)
    table.add_argument("--page", type=int, default=0)
    table.add_argument("--pdfium")
    table.add_argument("--json", action="store_true")
    for name in ("ocr-png", "ocr-pdf-page", "recognize-table-png", "recognize-wired-table-png"):
        command = commands.add_parser(name)
        command.add_argument("input")
        command.add_argument("--runtime", required=True)
        command.add_argument("--det-model", required=True)
        command.add_argument("--det-sha256", required=True)
        command.add_argument("--rec-model", required=True)
        command.add_argument("--rec-sha256", required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--min-confidence", type=float, default=0.0)
        if name in ("recognize-table-png", "recognize-wired-table-png"):
            command.add_argument("--structure-model", required=True)
            command.add_argument("--structure-sha256", required=True)
        if name == "recognize-table-png":
            command.add_argument(
                "--variant", choices=("slanet-plus", "slanet-1m"), default="slanet-plus"
            )
        if name == "recognize-wired-table-png":
            command.add_argument("--no-rotated-fix", action="store_true")
            command.add_argument("--no-blank-cell-rec", action="store_true")
        if name == "ocr-pdf-page":
            command.add_argument("--page", type=int, default=0)
            command.add_argument("--dpi", type=float, default=144.0)
            command.add_argument("--max-pixels", type=int, default=16000000)
            command.add_argument("--pdfium")
    classifier = commands.add_parser("classify-table-png")
    classifier.add_argument("input")
    classifier.add_argument("--runtime", required=True)
    classifier.add_argument("--model", required=True)
    classifier.add_argument("--sha256", required=True)
    classifier.add_argument("--output-type", choices=("probabilities", "logits"))
    classifier.add_argument("--variant", choices=("paddle", "qanything"), default="paddle")
    classifier.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    classifier.add_argument("--threads", type=int, default=1)
    layout = commands.add_parser("layout-png")
    layout.add_argument("input")
    layout.add_argument("--runtime", required=True)
    layout.add_argument("--model", required=True)
    layout.add_argument("--sha256", required=True)
    layout.add_argument("--threshold", type=float, default=0.5)
    layout.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    layout.add_argument("--threads", type=int, default=1)
    structure = commands.add_parser("table-structure-png")
    structure.add_argument("input")
    structure.add_argument("--runtime", required=True)
    structure.add_argument("--model", required=True)
    structure.add_argument("--sha256", required=True)
    structure.add_argument("--variant", choices=("slanet-plus", "slanet-1m"), default="slanet-plus")
    structure.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    structure.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "capabilities":
            print(json.dumps(capabilities(), ensure_ascii=False))
        elif args.command == "extract-text":
            output = parse_pdf_text(
                args.input,
                pdfium_library=args.pdfium,
                start_page=args.start,
                end_page=args.end,
                allow_partial=args.allow_partial,
            )
            print(json.dumps(asdict(output), ensure_ascii=False) if args.json else output.text)
        elif args.command == "recover-table":
            output = recover_pdf_table(
                args.input, args.bounds, page_index=args.page, pdfium_library=args.pdfium
            )
            print(json.dumps(asdict(output), ensure_ascii=False) if args.json else output.html)
        elif args.command == "layout-png":
            session = OnnxSession(
                args.runtime, args.model, args.sha256, device=args.device, threads=args.threads
            )
            print(
                json.dumps(
                    session.layout_png(args.input, threshold=args.threshold), ensure_ascii=False
                )
            )
        elif args.command == "classify-table-png":
            session = OnnxSession(
                args.runtime, args.model, args.sha256, device=args.device, threads=args.threads
            )
            print(
                json.dumps(
                    session.classify_table_png(
                        args.input, variant=args.variant, output_type=args.output_type
                    ),
                    ensure_ascii=False,
                )
            )
        elif args.command == "table-structure-png":
            session = OnnxSession(
                args.runtime, args.model, args.sha256, device=args.device, threads=args.threads
            )
            print(
                json.dumps(
                    session.table_structure_png(args.input, variant=args.variant),
                    ensure_ascii=False,
                )
            )
        else:
            detector = OnnxSession(
                args.runtime,
                args.det_model,
                args.det_sha256,
                device=args.device,
                threads=args.threads,
            )
            recognizer = OnnxSession(
                args.runtime,
                args.rec_model,
                args.rec_sha256,
                device=args.device,
                threads=args.threads,
            )
            options = dict(
                detector=detector, recognizer=recognizer, min_confidence=args.min_confidence
            )
            if args.command == "ocr-png":
                output = ocr_png(args.input, **options)
            elif args.command in ("recognize-table-png", "recognize-wired-table-png"):
                structure_model = OnnxSession(
                    args.runtime,
                    args.structure_model,
                    args.structure_sha256,
                    device=args.device,
                    threads=args.threads,
                )
                if args.command == "recognize-table-png":
                    output = recognize_table_png(
                        args.input, structure_model=structure_model, variant=args.variant, **options
                    )
                else:
                    output = recognize_wired_table_png(
                        args.input,
                        structure_model=structure_model,
                        rotated_fix=not args.no_rotated_fix,
                        blank_cell_rec=not args.no_blank_cell_rec,
                        **options,
                    )
            else:
                output = asdict(
                    ocr_pdf_page(
                        args.input,
                        page_index=args.page,
                        dpi=args.dpi,
                        max_pixels=args.max_pixels,
                        pdfium_library=args.pdfium,
                        **options,
                    )
                )
            print(json.dumps(output, ensure_ascii=False))
    except RapidDocError as error:
        print(
            json.dumps({"code": error.code, "message": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    except (OSError, TypeError, ValueError, OverflowError) as error:
        print(
            json.dumps({"code": "invalid_input", "message": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
