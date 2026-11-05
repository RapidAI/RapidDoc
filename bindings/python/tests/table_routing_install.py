"""真实 Paddle 分类驱动两个裁图分支；未选中的模型不调用，不吞掉坏结构。"""

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

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
cls = load("table-classification")
wired = load("wired-pipeline")
wireless = load("table-match")
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
classifier = OnnxSession(runtime, root / cls["model"], cls["sha256"])
secondary = OnnxSession(runtime, root / cls["qmodel"], cls["qsha256"])
unet = OnnxSession(runtime, root / wired["unet_model"], wired["unet_sha256"])
detector = OnnxSession(runtime, root / wired["det_model"], wired["det_sha256"])
recognizer = OnnxSession(runtime, root / wired["rec_model"], wired["rec_sha256"])
case = next(v for v in wired["cases"] if Path(v["input"]).name == "wired.png")
options = dict(
    classifier=classifier, detector=detector, recognizer=recognizer, cls_conf_threshold=0.0
)
result = recognize_table_auto_png(case["input"], wired_model=unet, wireless_model=None, **options)
assert result["route"] == result["classification"]["label"] == "wired"
assert result["html"] == case["html"] and result["matches"]["cell_lines"] == case["cell_lines"]
ensemble = recognize_table_auto_png(
    case["input"], wired_model=unet, wireless_model=None, secondary_classifier=secondary, **options
)
assert ensemble["route"] == "wired" and ensemble["html"] == case["html"]
assert len(ensemble["classification"]["members"]) == 2
for variant in ["slanet-plus", "slanet-1m"]:
    case = next(
        v
        for v in wireless["png_cases"]
        if v["variant"] == variant and Path(v["input"]).name == "wireless.png"
    )
    slanet = OnnxSession(runtime, root / case["model"], case["sha256"])
    result = recognize_table_auto_png(
        Path(case["input"]).read_bytes(),
        wired_model=None,
        wireless_model=slanet,
        wireless_variant=variant,
        **options,
    )
    assert result["route"] == result["classification"]["label"] == "wireless"
    assert result["matches"] == case["expected"]
    assert result["cell_texts"] == [
        " ".join(case["ocr_lines"][i]["text"] for i in indices)
        for indices in case["expected"]["cells"]
    ]
    ensemble = recognize_table_auto_png(
        case["input"],
        wired_model=None,
        wireless_model=slanet,
        wireless_variant=variant,
        secondary_classifier=secondary,
        **options,
    )
    assert ensemble["route"] == "wireless" and ensemble["html"] == result["html"]
    blank = next(
        v
        for v in wireless["png_cases"]
        if v["variant"] == variant and Path(v["input"]).name == "blank.png"
    )
    try:
        recognize_table_auto_png(
            blank["input"],
            wired_model=None,
            wireless_model=slanet,
            wireless_variant=variant,
            **options,
        )
    except RapidDocError as e:
        assert e.code == blank["expected_error"]
    else:
        raise AssertionError("bad blank structure became successful routed table")
print(
    "Installed wheel: Paddle and Paddle+Qanything→UNET/SLANet-plus/SLANet-1m routing, unused-model isolation and blank errors passed"
)
