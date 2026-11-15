"""Score the published Office outputs against the versioned fixture checklist."""
import argparse
from collections import defaultdict
import csv
from html import unescape
import json
from pathlib import Path
import re
from statistics import fmean
import unicodedata


def visible_text(markdown):
    text = unescape(re.sub(r"<[^>]+>", " ", markdown))
    text = re.sub(r"\\([*_`$\[\]])", r"\1", text)
    return " ".join(unicodedata.normalize("NFC", text).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing formal-anydoc and formal-rapid-office")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checklist = Path(__file__).resolve().parents[2] / "docs/benchmarks/firecrawl-office-checks.tsv"
    with checklist.open(encoding="utf-8", newline="") as stream:
        checks = list(csv.DictReader(stream, delimiter="\t"))
    results = []
    grouped = defaultdict(list)
    for engine, directory in (("AnyDoc", "formal-anydoc"), ("RapidDoc", "formal-rapid-office")):
        root = args.root / directory
        manifest = json.loads((root / "run.json").read_text("utf-8"))
        inputs = {row["file"]: row for row in manifest["documents"]}
        if set(inputs) != {check["file"] for check in checks}:
            raise ValueError(f"Checklist does not cover the input manifest for {engine}")
        for check in checks:
            file = check["file"]
            if inputs[file]["sha256"] != check["sha256"]:
                raise ValueError(f"Fixture differs from checklist: {file}")
            md = (root / (file + ".md")).read_text("utf-8")
            passed = (check["expected"] in visible_text(md) if check["match"] == "text"
                      else bool(re.search(check["expected"], md, flags=re.M)))
            row = dict(engine=engine, version=manifest["version"], **check, passed=int(passed))
            results.append(row)
            grouped[engine, Path(file).suffix[1:], file].append(int(passed))
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    for engine in ("AnyDoc", "RapidDoc"):
        scores = defaultdict(list)
        for (tool, fmt, _), passed in grouped.items():
            if tool == engine:
                scores[fmt].append(fmean(passed))
        formats = {fmt: 100 * fmean(values) for fmt, values in sorted(scores.items())}
        display = {fmt: round(value, 2) for fmt, value in formats.items()}
        checks_for_tool = [row for row in results if row["engine"] == engine]
        print(engine, display, "macro", round(fmean(formats.values()), 2),
              "checks", sum(row["passed"] for row in checks_for_tool), "/", len(checks_for_tool))


if __name__ == "__main__":
    main()
