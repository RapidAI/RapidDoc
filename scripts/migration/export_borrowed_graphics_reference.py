"""Capture an installed pre-change strong-graphics result from real raw pages."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
import argparse
import gzip
import json
from pathlib import Path


def main():
    from rapid_doc_rust import detect_native_strong_graphics
    from rapid_doc_rust._host_context import host_call
    detect = host_call(detect_native_strong_graphics)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.capture.read_bytes())
    sources = [data["pages"][index]["source"] for index in (0, 15, 53, 54)]
    del data
    cases = [dict(source=source, expected=detect(source)) for source in sources]
    args.output.write_bytes(gzip.compress(json.dumps(dict(cases=cases), ensure_ascii=False,
                                                     separators=(",", ":")).encode(), mtime=0))
    print("Captured installed strong graphics:", len(cases), "cases")


if __name__ == "__main__":
    main()
