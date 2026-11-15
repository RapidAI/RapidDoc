"""Run the published Rust SDK bindings against the Firecrawl parsers."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

OFFICE = {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["anydoc", "pdf-inspector", "rapid-office",
                                          "rapid-default", "rapid-rule"])
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--exceptional", action="store_true", help="Run only abuse and malformed Office fixtures")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    office = args.engine in {"anydoc", "rapid-office"}
    suffixes = OFFICE if office else {".pdf"}
    candidates = sorted(p for p in args.inputs.rglob("*") if p.suffix.lower() in suffixes)
    sources = {}
    duplicates = {}
    for path in candidates:
        exceptional = path.relative_to(args.inputs).parts[0] in {"abuse", "malformed"}
        if office and exceptional != args.exceptional:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if office and digest in sources:
            duplicates[path.relative_to(args.inputs).as_posix()] = sources[digest][0]
            continue
        name = path.relative_to(args.inputs).as_posix()
        sources[digest if office else name] = (name, path, digest)
    if not sources:
        parser.error("no matching inputs")
    if args.engine == "anydoc":
        import anydoc

        def convert(data):
            return anydoc.to_markdown_bytes(data)
        distribution = "firecrawl-anydoc"
    elif args.engine == "pdf-inspector":
        import pdf_inspector

        def convert(data):
            return pdf_inspector.process_pdf_bytes(data).markdown
        distribution = "pdf-inspector"
    else:
        from loguru import logger
        from rapid_doc_rust import RapidDoc

        logger.remove()
        engine = RapidDoc(parse_method="txt", formula_enable=False, table_enable=True,
                          layout_config={"model_type": "docvortex_rule" if
                                         args.engine == "rapid-rule" else "pp_doclayoutv3"})

        def convert(data):
            return engine(data, f_dump_middle_json=False, f_dump_content_list=False).markdown
        distribution = "rapid-doc-rust"
    args.output.mkdir(parents=True, exist_ok=True)
    documents = []
    for index, (name, path, digest) in enumerate(sources.values(), 1):
        row = {"file": name, "sha256": digest, "seconds": []}
        try:
            data = path.read_bytes()
            if index == 1:
                started = perf_counter()
                try:
                    convert(data)
                except Exception as exc:
                    row["warmup_error"] = f"{type(exc).__name__}: {exc}"
                row["warmup_seconds"] = perf_counter() - started
            for _ in range(args.repeat):
                started = perf_counter()
                markdown = convert(data)
                row["seconds"].append(perf_counter() - started)
            if not markdown or not markdown.strip():
                raise ValueError("Empty Markdown")
            row["median_seconds"] = median(row["seconds"])
            row["status"] = "ok"
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            markdown = ""
        markdown_root = args.output if office else args.output / "markdown"
        destination = (markdown_root / name).with_suffix(path.suffix + ".md" if office else ".md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
        documents.append(row)
        print(f"{index}/{len(sources)} {row['status']} {name} {row.get('error', '')}", flush=True)
    (args.output / "run.json").write_text(json.dumps({
        "engine": args.engine, "version": importlib.metadata.version(distribution),
        "platform": platform.platform(), "python": platform.python_version(),
        "repeat": args.repeat, "duplicates": duplicates, "documents": documents,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
