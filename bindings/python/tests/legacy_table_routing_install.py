"""独立wheel旧原HTML选优、阈值/调用顺序和实际CPU双模型路由。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from pathlib import Path
from rapid_doc_rust import (
    OnnxSession,
    select_legacy_table_candidate,
    recognize_legacy_table_auto_png,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/legacy-table-routing.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
for c in f["selection"]:
    r = select_legacy_table_candidate(
        c["texts"], wired_html=c["wired_html"], wireless_html=c["wireless_html"]
    )
    for k in ("html", "wired", "wireless"):
        assert r[k] == c[k], (k, r, c)


class Classifier:
    def __init__(self, label, score):
        self.label = label
        self.score = score

    def classify_table_png(self, *a, **kw):
        return dict(label=self.label, score=self.score)


unet = json.loads((root / "build/migration/legacy-unet.json").read_text(encoding="utf-8"))
source = root / unet["real"][0]["source"]
for c in f["policy"]:
    calls = []

    class Wired:
        def table_cells_png(self, *a, **kw):
            calls.append("wired")
            return dict(
                polygons=[[[0, 0], [0, 30], [40, 30], [40, 0]]],
                topology=dict(logic_points=[[0, 0, 0, 0]]),
            )

    class Wireless:
        def table_structure_legacy_png(self, *a, **kw):
            calls.append("wireless")
            return dict(
                tokens=["<table>", "<tr>", "<td></td>", "</tr>", "</table>"],
                cell_bboxes=[[0, 0, 40, 30]],
            )

    r = recognize_legacy_table_auto_png(
        source,
        c["ocr_result"],
        classifier=Classifier(c["label"], c["score"]),
        wired_model=Wired(),
        wireless_model=Wireless(),
        use_compare_table=c["compare"],
    )
    assert calls == c["calls"], (c, calls)
    # 固定响应候选只用于调用顺序；实际候选原HTML精确对照见下方真实模型。
    assert r["classification"]["label"] == c["label"] and r["classification"]["score"] == c["score"]
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
cls = f["classifiers"]
classifier = OnnxSession(runtime, root / cls["model"], cls["sha256"])
secondary = OnnxSession(runtime, root / cls["qmodel"], cls["qsha256"])
wired = OnnxSession(runtime, root / f["unet_model"], f["unet_sha256"])
wireless = OnnxSession(runtime, root / f["slanet_model"], f["slanet_sha256"])
forced = json.loads(
    (root / "build/migration/legacy-table-routing-compare.json").read_text(encoding="utf-8")
)
for compare, cases in [(False, f["real"]), (True, forced["real"])]:
    for c in cases:
        before = copy.deepcopy(c["ocr_result"])
        r = recognize_legacy_table_auto_png(
            root / c["source"],
            c["ocr_result"],
            classifier=classifier,
            secondary_classifier=secondary,
            wired_model=wired,
            wireless_model=wireless,
            use_compare_table=compare,
        )
        assert r["html"] == c["html"], (c["source"], r["html"], c["html"])
        assert c["ocr_result"] == before
        if compare:
            assert set(r["candidates"]) == {"wired", "wireless"} and r["selection"] is not None


class Empty:
    def detect_table_png(self, *a):
        return []


assert recognize_legacy_table_auto_png(source, None, detector=Empty())["classification"] is None
for bad in [True, float("nan"), -1]:
    try:
        recognize_legacy_table_auto_png(source, None, cls_conf_threshold=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("bad classifier threshold accepted")
print(
    "Installed wheel: 237 old raw HTML selections/metrics, 12 threshold/call-order policies and 10 real CPU Paddle+Qanything/UNET/SLANet routes (5 forced comparisons) passed"
)
