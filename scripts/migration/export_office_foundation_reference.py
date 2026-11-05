"""Capture existing Python OPC helpers and OLE streams for the Rust foundation."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office import opc
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xlsx.ooxml_ole import _resolve_internal_target
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy.ole import BoundedOleReader
    reference = ROOT / "tests/fixtures/office_rust_reference_068"
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for rels in ("_rels/.rels", "word/_rels/document.xml.rels", "custom/a/_rels/root.xml.rels",
                 "_rels/root.xml.rels", "invalid", "word\\_rels\\root.xml.rels"):
        for target in (None, "", "a.xml", "../a.xml", "../../a.xml", "..", "../..", "/a/../b.xml",
                       "a\\b.xml", "a//b.xml", "./a.xml", " /a.xml ", "https://example.test/a"):
            cases.append(dict(rels=rels, target=target,
                              expected=opc._resolve_relationship_target(rels, target)))
    parts = ["", ".", "word/document.xml", "main.xml", "word/", "custom\\a.xml", "/a.xml"]
    embedded = [dict(source=source, target=target, expected=_resolve_internal_target(source, target))
                for source in ("xl/worksheets/sheet1.xml", "main.xml")
                for target in ("../media/a.png", "../../../outside", "..", "a\\b", " /xl/media/a.png ", "\x1ca.xml\x1f")]
    ole = []
    for case in manifest["cases"]:
        if case["format"] not in {"doc", "xls", "ppt"}:
            continue
        data = (ROOT / case["input"]).read_bytes()
        with BoundedOleReader(data) as reader:
            streams = [dict(name=name, sha256=hashlib.sha256(reader.read_stream(name)).hexdigest())
                       for name in reader.stream_names()]
        ole.append(dict(input=case["input"], format=case["format"], sha256=case["input_sha256"], streams=streams))
    output = dict(relationship_targets=cases, embedded_targets=embedded,
                  relationship_members=[dict(part=part, expected=opc._part_relationships_member(part)) for part in parts],
                  source_directories=[dict(rels=rels, expected=opc.relationship_source_base_dir(rels))
                                      for rels in {case["rels"] for case in cases}],
                  ole=ole,
                  casefold=[dict(text=text, expected=text.casefold()) for text in ("Straße", "İΣς", "ﬃＫK", "WordDocument")])
    output["source_directories"].sort(key=lambda row: row["rels"])
    (reference / "foundation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Foundation references: {len(cases)} OPC targets, {len(embedded)} embedded targets, {len(ole)} OLE files")


if __name__ == "__main__":
    main()
