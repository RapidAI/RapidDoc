"""Generate OpenDataLoader benchmark predictions with local RapidDoc.

This script is intentionally kept in the RapidDoc repo so it can import the
unreleased local package, while writing files in the layout expected by
opendataloader-bench:

    prediction/rapiddoc/markdown/<document_id>.md
    prediction/rapiddoc/summary.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()

DEFAULT_ENGINE_NAME = "rapiddoc"
RULE_TABLE_ENGINE_NAME = "rapiddoc-rule_table"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_ROOT = PROJECT_ROOT.parent / "opendataloader-bench-main"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

layout_config = {
    "markdown_ignore_labels": [],
}
table_config = {
    # "use_rule_table": True,
}

_engines = {}


def _effective_use_rule_table(use_rule_table: Optional[bool] = None) -> bool:
    if use_rule_table is not None:
        return use_rule_table
    return bool(table_config.get("use_rule_table", False))


def _default_engine_name(use_rule_table: Optional[bool] = None) -> str:
    if _effective_use_rule_table(use_rule_table):
        return RULE_TABLE_ENGINE_NAME
    return DEFAULT_ENGINE_NAME


def _get_engine(use_rule_table: Optional[bool] = None):
    use_rule_table = _effective_use_rule_table(use_rule_table)
    if use_rule_table not in _engines:
        from rapid_doc import RapidDoc

        effective_table_config = dict(table_config)
        # effective_table_config["use_rule_table"] = use_rule_table
        _engines[use_rule_table] = RapidDoc(
            table_config=effective_table_config,
            layout_config=layout_config,
        )
    return _engines[use_rule_table]


def _get_rapiddoc_version() -> str:
    try:
        from rapid_doc.version import __version__
    except Exception:  # pragma: no cover - best effort metadata only
        return "local"
    return __version__


def to_markdown(
    doc_paths: Sequence[Path | str],
    _,
    output_dir: Path | str,
    use_rule_table: Optional[bool] = None,
) -> None:
    """Convert PDFs to Markdown using the OpenDataLoader bench parser contract."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(doc_path) for doc_path in doc_paths]
    outputs = _get_engine(use_rule_table)(paths)
    for doc_path, output in zip(paths, outputs):
        output_file = output_dir / f"{doc_path.stem}.md"
        output_file.write_text(output.markdown, encoding="utf-8", errors="replace")


def _iter_pdf_paths(input_dir: Path, doc_id: Optional[str]) -> list[Path]:
    if doc_id:
        pdf_path = input_dir / f"{doc_id.strip()}.pdf"
        if not pdf_path.is_file():
            raise FileNotFoundError(f"{pdf_path} not found")
        return [pdf_path]

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")
    return pdf_paths


def _processor_name() -> str:
    processor = platform.processor()
    return processor or platform.machine() or "unknown"


def export_benchmark_predictions(
    input_dir: Path,
    prediction_root: Path,
    doc_id: Optional[str] = None,
    engine_name: Optional[str] = None,
    use_rule_table: Optional[bool] = None,
) -> Path:
    """Write RapidDoc predictions and summary under a bench prediction root."""
    use_rule_table = _effective_use_rule_table(use_rule_table)
    engine_name = engine_name or _default_engine_name(use_rule_table)
    doc_paths = _iter_pdf_paths(input_dir, doc_id)
    engine_dir = prediction_root / engine_name
    markdown_dir = engine_dir / "markdown"

    start_time = time.time()
    to_markdown(doc_paths, input_dir, markdown_dir, use_rule_table=use_rule_table)
    total_elapsed = time.time() - start_time

    summary = {
        "engine_name": engine_name,
        "engine_version": _get_rapiddoc_version(),
        "table_config": {
            "use_rule_table": use_rule_table,
        },
        "processor": _processor_name(),
        "document_count": len(doc_paths),
        "total_elapsed": total_elapsed,
        "elapsed_per_doc": total_elapsed / len(doc_paths),
        "date": time.strftime("%Y-%m-%d"),
    }
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "summary.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return engine_dir


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local RapidDoc markdown predictions for opendataloader-bench."
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help="Path to opendataloader-bench-main.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="PDF directory. Defaults to <bench-root>/pdfs.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=None,
        help="Prediction root. Defaults to <bench-root>/prediction.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Only process one document id, for example 01030000000001.",
    )
    parser.add_argument(
        "--use-rule-table",
        dest="use_rule_table",
        action="store_true",
        default=None,
        help=f"Enable RapidDoc rule table mode and default to {RULE_TABLE_ENGINE_NAME}.",
    )
    parser.add_argument(
        "--no-rule-table",
        dest="use_rule_table",
        action="store_false",
        help=f"Disable RapidDoc rule table mode and default to {DEFAULT_ENGINE_NAME}.",
    )
    parser.add_argument(
        "--engine-name",
        type=str,
        default=None,
        help="Prediction engine directory name. Defaults to rapiddoc or rapiddoc-rule_table.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    bench_root = args.bench_root.resolve()
    # input_dir = (args.input_dir or bench_root / "pdfs").resolve()
    input_dir = Path(r'D:\CodeProjects\doc\RapidAI\opendataloader-bench-main\pdfs')
    prediction_root = (args.prediction_root or bench_root / "prediction").resolve()
    use_rule_table = _effective_use_rule_table(args.use_rule_table)
    engine_name = args.engine_name or _default_engine_name(use_rule_table)

    engine_dir = export_benchmark_predictions(
        input_dir,
        prediction_root,
        args.doc_id,
        engine_name=engine_name,
        use_rule_table=use_rule_table,
    )
    print(f"Wrote RapidDoc predictions to {engine_dir}")
    print(
        "Evaluate with: "
        f"Set-Location {bench_root}; uv run src/evaluator.py --engine {engine_name}"
    )


if __name__ == "__main__":
    main()
