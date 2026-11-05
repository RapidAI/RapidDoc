"""Capture candidate merge reference inputs; never overwrite an existing fixture."""
import argparse
import gzip
import importlib.util
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def encode(candidate):
    value = asdict(candidate)
    value["line_indices"] = sorted(value["line_indices"])
    for annotation in value["annotations"]:
        annotation["line_indices"] = sorted(annotation["line_indices"])
        annotation["line_bboxes"] = list(map(list, annotation["line_bboxes"].items()))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_rules
    from rapid_doc._vendor.docvortex_core.analyzers.native.models import PdfModel
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from pypdf import PdfReader, PdfWriter
    from loguru import logger
    logger.remove()
    spec = importlib.util.spec_from_file_location("caption_grid_cases", ROOT / "tests/native_pdf/test_caption_grid_merge.py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    cases = []

    def add(label, candidates):
        request = dict(candidates=[encode(c) for c in candidates])
        result = table_rules._merge_table_candidates(candidates)
        cases.append(dict(label=label, request=request, expected=dict(candidates=[encode(c) for c in result])))

    for angle in (0, 90, 180, 270):
        for kind in ("caption", "footnote"):
            target, weak = test.candidates(angle, kind)
            add(f"{kind}:{angle}", [weak, target])
    for reason in ("core_missing", "core_member", "inside_core", "outside_bounds", "missing_box"):
        target, weak = test.candidates(core=reason != "core_missing")
        annotation = weak.annotations[0]
        index = {"core_member": 1, "inside_core": 10, "outside_bounds": 11, "missing_box": 13}.get(reason, 9)
        annotation.line_indices = {index}
        annotation.line_bboxes = {i: b for i, b in annotation.line_bboxes.items() if i == index}
        add(reason, [target, weak])

    original = table_rules._merge_table_candidates
    real_index = 0

    def capture(candidates):
        nonlocal real_index
        materialized = [c.materialize() if isinstance(c, table_rules._RuleCandidateDraft) else c for c in candidates]
        request = dict(candidates=[encode(c) for c in materialized])
        result = original(materialized)
        cases.append(dict(label=f"actual-page-host-merge:{real_index}", request=request,
                          expected=dict(candidates=[encode(c) for c in result])))
        real_index += 1
        return result

    writer = PdfWriter()
    writer.add_page(PdfReader(ROOT / "tests/fixtures/upstream_068/deep_structure_catalog.pdf").pages[53])
    payload = io.BytesIO()
    writer.write(payload)
    table_rules._merge_table_candidates = capture
    try:
        with PDFDocument(payload.getvalue()) as document:
            PdfModel().predict(document)
    finally:
        table_rules._merge_table_candidates = original
    data = json.dumps(dict(cases=cases), ensure_ascii=True, separators=(",", ":")).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(data, mtime=0))
    print(f"Saved {len(cases)} candidate merge cases: {args.output}")


if __name__ == "__main__":
    main()
