"""Freeze complete table candidates from the installed pre-cache implementation."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
import argparse
import gzip
import json
from pathlib import Path


def main():
    from rapid_doc_rust import legacy_pipeline_json
    from rapid_doc_rust._host_context import host_call
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    @host_call
    def detect(request):
        return json.loads(legacy_pipeline_json("detect_native_table_candidates", json.dumps(request, allow_nan=False)))

    sources = json.loads(gzip.decompress(args.sources.read_bytes()))["cases"]
    cases = []
    for item in sources:
        request = dict(source=item["source"], excluded_bboxes=[])
        cases.append(dict(request=request, expected=detect(request)))
    args.output.write_bytes(gzip.compress(json.dumps(dict(cases=cases), ensure_ascii=False,
                                                     separators=(",", ":")).encode(), mtime=0))
    print("Captured complete table candidates:", len(cases), "cases")


if __name__ == "__main__":
    main()
