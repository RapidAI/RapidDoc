from pathlib import Path
import json, hashlib, sys
from rapid_doc_rust._native import draw_table_lines_bytes

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-draw.json").read_text())
for case in fixture["cases"]:
    result = draw_table_lines_bytes(case["lines"], case["width"], case["height"])
    if hashlib.sha256(result).hexdigest() != case["sha256"]:
        expected = Path(case["expected"]).read_bytes()
        diff = [
            (i % case["width"], i // case["width"], a, b)
            for i, (a, b) in enumerate(zip(result, expected))
            if a != b
        ]
        raise AssertionError((case["id"], len(diff), diff[:20]))
print(f"Installed wheel: {len(fixture['cases'])} thickness-2 LINE_AA rasters pixel-exact")
