"""Compare selected installed Rust results with the frozen Python baseline."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.migration.accept_pdf_delivery import comparison, snapshot
from rapid_doc_rust import RapidDoc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "build/pdf-delivery-acceptance-20261030")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.baseline / "manifest.json").read_text("utf-8"))
    available = {Path(c["source"]).stem for c in manifest["cases"]}
    if set(args.only) - available:
        parser.error(f"Unknown cases: {set(args.only) - available}")
    engine = RapidDoc(formula_config={"model_dir_or_path": manifest["formula_model"]})
    reports = []
    for case in manifest["cases"]:
        if Path(case["source"]).stem not in args.only:
            continue
        folder = args.output / case["id"]
        folder.mkdir()
        assert hashlib.sha256((ROOT / case["source"]).read_bytes()).hexdigest() == case["sha256"]
        if args.trace:
            os.environ["RAPIDDOC_PARITY_TRACE"] = str(folder.resolve())
        result = engine(ROOT / case["source"], output_dir=folder / "exports")
        actual = snapshot(result)
        (folder / "snapshot.json").write_text(json.dumps(actual, ensure_ascii=False), "utf-8")
        expected = json.loads((args.baseline / case["id"] / "python/cold.json").read_text("utf-8"))
        report = comparison(expected, actual)
        (folder / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        exact = all(report[field]["exact"] for field in ("markdown", "content_list_json", "middle_json"))
        exact &= report["assets"]["exact_pixels_and_names"]
        reports.append(dict(case=case["id"], exact=exact))
        print("PARITY", reports[-1], flush=True)
    (args.output / "summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), "utf-8")
    return 0 if all(row["exact"] for row in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
