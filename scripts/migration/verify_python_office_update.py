"""Capture or strictly compare the current Python Office reference outputs."""
import argparse
import json
from pathlib import Path
from common import ROOT, load_manifest, parse_case, write_json
from compare_results import differences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--before", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [c for c in load_manifest(ROOT / "tests/migration/corpus_manifest.json")
             if c["format"] in {"doc", "docx", "xls", "xlsx", "ppt", "pptx"}]
    rows = []
    for case in cases:
        result, timings = parse_case(case)
        write_json(args.output / (case["id"] + ".json"), result)
        row = dict(case=case["id"], format=case["format"], timings=timings)
        if args.before:
            old = json.loads((args.before / (case["id"] + ".json")).read_text("utf-8"))
            changes = list(differences(old, result))
            row.update(exact=not changes, differences=changes)
        rows.append(row)
        print(row["case"], row.get("exact", "captured"), flush=True)
    write_json(args.output / "summary.json", rows)
    return 1 if any(r.get("exact") is False for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
