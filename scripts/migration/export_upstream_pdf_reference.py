"""Replay upstream PDF regression inputs with RapidDoc's updated Python reference.

Capture inputs during the upstream tests, then recompute expectations after pytest
has restored all monkeypatches. Never replace an existing reference.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {field.name: clean(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if hasattr(value, "bbox"):
        return list(value.bbox)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pytest
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import engine
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline import materialize
    from loguru import logger
    logger.remove()
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_materialization, table_text_styles
    tables, inline, renders = {}, {}, {}
    omitted = []

    def remember(target, value):
        try:
            encoded = json.dumps(clean(value), ensure_ascii=True, allow_nan=False, sort_keys=True)
        except ValueError:
            omitted.append("Non-finite metadata: retained in Python validation tests; not JSON-serializable")
            return
        target.setdefault(hashlib.sha256(encoded.encode()).hexdigest(), copy.deepcopy(value))

    evaluate = engine._evaluate_native_pdf_table
    apply = materialize.apply_pdf_inline_evidence
    render = table_text_styles.render_native_table_html_with_scripts

    def table_input(value):
        remember(tables, value)
        return evaluate(value)

    def inline_input(blocks, links, styles, scripts, size, **kwargs):
        remember(inline, dict(blocks=blocks, links=links, styles=styles, scripts=scripts, size=size))
        return apply(blocks, links, styles, scripts, size, **kwargs)

    def render_input(result, table_input, tight_bboxes, origins, **kwargs):
        remember(renders, (result, table_input, tight_bboxes, origins, kwargs.get("preserve_font_styles", False)))
        return render(result, table_input, tight_bboxes, origins, **kwargs)

    table_text_styles.render_native_table_html_with_scripts = render_input
    table_materialization.render_native_table_html_with_scripts = render_input
    engine._evaluate_native_pdf_table = table_input
    materialize.apply_pdf_inline_evidence = inline_input
    try:
        result = pytest.main([
            "-q", *map(str, sorted((ROOT / "tests/native_pdf").glob("test_upstream_068_*.py"))),
            "-k", "not full_catalog",
        ])
    finally:
        engine._evaluate_native_pdf_table = evaluate
        materialize.apply_pdf_inline_evidence = apply
        table_text_styles.render_native_table_html_with_scripts = render
        table_materialization.render_native_table_html_with_scripts = render
    if result:
        raise RuntimeError(f"Upstream regression tests failed: {result}")
    cases = []
    for digest, value in tables.items():
        evaluation = evaluate(value)
        recovered = engine.recover_native_pdf_table(value)
        cases.append(dict(kind="table", digest=digest, request=dict(table_input=clean(value)), expected=clean(dict(
            evaluation=evaluation, result=recovered, diagnostics=engine.diagnose_native_pdf_table(value)))))
    for digest, value in inline.items():
        blocks, diagnostics = copy.deepcopy(value["blocks"]), []
        apply(blocks, value["links"], value["styles"], value["scripts"], value["size"], materialized_diagnostics=diagnostics)
        request = dict(blocks=clean(value["blocks"]), link_lines=clean(value["links"]), style_lines=clean(value["styles"]),
                       script_lines=clean(value["scripts"]), page_size=clean(value["size"]))
        cases.append(dict(kind="inline", digest=digest, request=request,
                          expected=clean(dict(blocks=blocks, materialized_ranges=diagnostics))))
    for digest, (result, table_input, tight, origins, preserve) in renders.items():
        request = clean(dict(result=result, table_input=table_input, tight_bboxes=tight, origins=origins, preserve_font_styles=preserve))
        cases.append(dict(kind="render", digest=digest, request=request,
                          expected=render(result, table_input, tight, origins, preserve_font_styles=preserve)))
    source_files = [*sorted((ROOT / "rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery").glob("*.py")),
                    *sorted((ROOT / "rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline").glob("*.py"))]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(dict(sources=sources, cases=cases, omitted=omitted), ensure_ascii=True, indent=2).encode())
    print(f"Saved {len(tables)} table / {len(inline)} inline / {len(renders)} styled table inputs: {args.output}")


if __name__ == "__main__":
    main()
