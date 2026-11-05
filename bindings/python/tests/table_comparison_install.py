"""真实分类/模型输出的低分/显式比较路由，共用一次 OCR，并保留阈值边界。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
import json, sys
from rapid_doc_rust import OnnxSession, recognize_table_auto_png, RapidDocError
from rapid_doc_rust._native import decode_table_classification_json

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
cls = load("table-classification")
wired = load("wired-pipeline")
wireless = load("table-match")
quality = load("table-quality")
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
classifier = OnnxSession(runtime, root / cls["model"], cls["sha256"])
secondary = OnnxSession(runtime, root / cls["qmodel"], cls["qsha256"])
unet = OnnxSession(runtime, root / wired["unet_model"], wired["unet_sha256"])
detector = OnnxSession(runtime, root / wired["det_model"], wired["det_sha256"])
recognizer = OnnxSession(runtime, root / wired["rec_model"], wired["rec_sha256"])


class CountedDetector:
    def __init__(self):
        self.calls = 0

    @property
    def _native(self):
        self.calls += 1
        return detector._native


counted = CountedDetector()
options = dict(classifier=classifier, wired_model=unet, detector=counted, recognizer=recognizer)
for variant in ["slanet-plus", "slanet-1m"]:
    case = next(v for v in wireless["png_cases"] if v["variant"] == variant)
    slanet = OnnxSession(runtime, root / case["model"], case["sha256"])
    for name in ["wired", "wireless"]:
        case = next(v for v in wired["cases"] if Path(v["input"]).name == name + ".png")
        expected = next(v for v in quality["cases"] if v["id"] == f"actual-{variant}-{name}")
        counted.calls = 0
        # wired 的高分场景显式比较，wireless 的低分场景使用默认阈值触发。
        result = recognize_table_auto_png(
            case["input"],
            wireless_model=slanet,
            wireless_variant=variant,
            use_compare_table=name == "wired",
            **options,
        )
        assert counted.calls == 1, "comparison repeated detector inference"
        assert result["html"] == expected["expected_html"], (
            variant,
            name,
            result["comparison"],
            result["html"],
            expected["expected_html"],
        )
        assert result["route"] == result["comparison"]["route"] == expected["expected_route"]
        assert (
            result["comparison"]["wired"] == expected["wired"]
            and result["comparison"]["wireless"] == expected["wireless"]
        )
    if variant == "slanet-plus":
        case = wired["cases"][0]
        counted.calls = 0
        result = recognize_table_auto_png(
            case["input"], wireless_model=slanet, secondary_classifier=secondary, **options
        )
        assert result["comparison"] is not None and counted.calls == 1
        assert len(result["classification"]["members"]) == 2
        assert result["classification"]["score"] < 0.9
case = wired["cases"][0]
counted.calls = 0
high = recognize_table_auto_png(case["input"], wireless_model=None, **options)
assert high["comparison"] is None and high["route"] == "wired" and counted.calls == 1
counted.calls = 0
try:
    recognize_table_auto_png(case["input"], wireless_model=None, use_compare_table=True, **options)
except ValueError as error:
    assert "models" in str(error)
else:
    raise AssertionError("comparison did not require alternate model")
assert counted.calls == 0


class Fixed:
    def __init__(self, result):
        self.result = result

    def classify_table_png(self, *args, **kwargs):
        return self.result


score = json.loads(decode_table_classification_json([0.9, 0.1], "probabilities"))
assert score["score"] < 0.9, "FP32 score was rounded across the comparison threshold"
boundary = dict(options, classifier=Fixed(score))
counted.calls = 0
try:
    recognize_table_auto_png(case["input"], wireless_model=None, **boundary)
except ValueError:
    pass
else:
    raise AssertionError("FP32 0.9 score should trigger comparison")
assert counted.calls == 0
boundary["classifier"] = Fixed(dict(label="wired", score=0.9))
equal = recognize_table_auto_png(case["input"], wireless_model=None, **boundary)
assert equal["comparison"] is None, "comparison used <= instead of <"
for value in [float("nan"), -0.1, 1.1]:
    try:
        recognize_table_auto_png(
            b"",
            classifier=None,
            wired_model=None,
            wireless_model=None,
            detector=None,
            recognizer=None,
            cls_conf_threshold=value,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid classification threshold accepted")
print(
    "Installed wheel: 4 real two-model comparisons, default dual-classifier comparison, single-OCR sharing, high/equal/FP32 threshold boundaries and missing-model guards passed"
)
