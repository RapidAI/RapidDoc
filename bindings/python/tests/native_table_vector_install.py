"""独立wheel物理证据和规范轨道，与未改动旧vector.py参考逐项对照。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    prepare_native_table_rule_evidence,
    canonicalize_native_table_tracks,
    RapidDocError,
    capabilities,
)
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f["cases"]


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(pair):
    fn, c = pair
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    same(fn(**r), c["expected"])
    same(r, before, "ownership")


evidence = load("native-table-vector")
tracks = load("native-table-tracks") + load("native-table-track-ties")
cases = [(prepare_native_table_rule_evidence, c) for c in evidence] + [
    (canonicalize_native_table_tracks, c) for c in tracks
]
for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::25]))


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid physical evidence accepted")


r = copy.deepcopy(evidence[0]["request"])
r["table_input"]["drawing_lines"] *= 1000
fails(lambda: prepare_native_table_rule_evidence(**r))
r = copy.deepcopy(evidence[0]["request"])
r["snap_tolerance"] = -1.0
fails(lambda: prepare_native_table_rule_evidence(**r))
r = copy.deepcopy(evidence[0]["request"])
r["join_gap"] = float("nan")
fails(lambda: prepare_native_table_rule_evidence(**r))
r = copy.deepcopy(tracks[0]["request"])
r["positions"] = [0.0] * 201
fails(lambda: canonicalize_native_table_tracks(**r))
r = copy.deepcopy(tracks[0]["request"])
r["glyph_centers"] = [0.0] * 3001
fails(lambda: canonicalize_native_table_tracks(**r))
# 给定合法几何但二次工作超过共享预算时必须显式拒绝；不返回部分轨道。
r = copy.deepcopy(evidence[0]["request"])
r["table_input"]["drawing_lines"] = []
r["table_input"]["rectangles"] = [
    dict(bbox=[20.0, 30.0, 30.0, 40.0], segment_count=5, fill_visible=True, stroke_visible=False)
] * 5000
r["snap_tolerance"] = 0.5
r["evidence_halo"] = 1.0
r["include_rectangles"] = True
fails(lambda: prepare_native_table_rule_evidence(**r))
assert (
    capabilities()["native_table_vector_evidence"]
    and not capabilities()["legacy_native_table_recovery"]
)
for name in ["PYTHON-SET.txt", "MIGRATION-NOTICES.txt"]:
    reference = (
        root / "licenses" / name
        if name == "MIGRATION-NOTICES.txt"
        else root / "rust/rapiddoc/licenses" / name
    )
    assert (Path(sdk.__file__).parent / "licenses" / name).read_bytes() == reference.read_bytes()
print(
    f"Installed wheel: {len(evidence)} old physical evidence +{len(tracks)} canonical tracks; exact values/types, ownership/concurrency/resource guards; full vector candidate/recovery not implemented"
)
