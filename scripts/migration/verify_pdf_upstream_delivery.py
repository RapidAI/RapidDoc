"""Compare updated Python/Rust PDF output without replacing the accepted baseline."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.migration.accept_pdf_delivery import comparison, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("python", "rust"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "build/pdf-delivery-acceptance-20261030/manifest.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text("utf-8"))
    selected = [c for c in manifest["cases"] if not args.only or Path(c["source"]).stem in args.only]
    if args.only and set(args.only) != {Path(c["source"]).stem for c in selected}:
        raise ValueError("Unknown PDF case")
    if args.engine == "python":
        from rapid_doc import RapidDoc
    else:
        from rapid_doc_rust import RapidDoc
    engine = RapidDoc(formula_config={"model_dir_or_path": manifest["formula_model"]})
    reports = []
    for case in selected:
        source = ROOT / case["source"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
        folder = args.output / case["id"]
        folder.mkdir()
        actual = snapshot(engine(source, output_dir=folder / "exports"))
        (folder / "snapshot.json").write_bytes(json.dumps(actual, ensure_ascii=False).encode())
        row = dict(case=case["id"], pages=len(actual["middle_json"]["pdf_info"]))
        if args.reference:
            expected = json.loads((args.reference / case["id"] / "snapshot.json").read_text("utf-8"))
            report = comparison(expected, actual)
            (folder / "comparison.json").write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode())
            row["exact"] = (all(report[f]["exact"] for f in ("markdown", "content_list_json", "middle_json"))
                            and report["assets"]["exact_pixels_and_names"])
            row["encoded_assets_exact"] = all(a["encoded_sha256"] == b["encoded_sha256"]
                                               for a, b in zip(expected["assets"], actual["assets"]))
        reports.append(row)
        (args.output / "summary.json").write_bytes(json.dumps(reports, ensure_ascii=False, indent=2).encode())
        print("UPSTREAM_DELIVERY", args.engine, row, flush=True)
    return 1 if any(row.get("exact") is False for row in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
