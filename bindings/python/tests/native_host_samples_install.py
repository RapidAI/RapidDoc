"""Two user PDFs: original native host, real CPU formula/OCR, exact Markdown and pixels."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import ast, base64, copy, hashlib, json, re, struct, sys, zlib
from pathlib import Path
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-host-samples.json").read_text(encoding="utf-8"))


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, str(a)[:240], str(b)[:240])


tree = ast.parse(
    (root / "bindings/python/tests/native_table_media_png_install.py").read_text(encoding="utf-8")
)
exec(
    compile(
        ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pixels"],
            type_ignores=[],
        ),
        "verified_png_decoder",
        "exec",
    ),
    globals(),
)
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
m = f["models"]
sessions = {
    k: sdk.OnnxSession(runtime, root / m[k + "_model"], m[k + "_sha256"])
    for k in ["det", "rec", "orientation", "table"]
}
formula = json.loads((root / "build/migration/formula-decode.json").read_text(encoding="utf-8"))[
    "model"
]
formula = sdk.OnnxSession(runtime, root / formula["path"], formula["sha256"])
for i, c in enumerate(f["cases"]):
    path = root / c["source"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c["source_sha256"]
    out = sdk.parse_pdf_models(
        path,
        docvortex_mode="rule",
        native_only=True,
        detector=sessions["det"],
        recognizer=sessions["rec"],
        orientation=sessions["orientation"],
        table_model=sessions["table"],
        formula_model=formula,
        dpi=c["dpi"],
    )
    metadata = {k: v for k, v in out.items() if k != "assets"}
    (root / f"build/native-host-samples-actual-{i}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    same(out["pages"], c["expected"]["pages"], path.name + "/pages")
    same(out["markdown"], c["expected"]["markdown"], path.name + "/markdown")
    same(out["content_list"], c["expected"]["content_list"], path.name + "/content_list")
    same(
        [dict(name=n, **pixels(b)) for n, b in out["assets"].items()],
        c["assets"],
        path.name + "/assets",
    )
    destination = root / "build/rust-native-host-preview" / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "document.md").write_text(out["markdown"], encoding="utf-8")
    (destination / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, data in out["assets"].items():
        target = destination / name
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert all(r["method"] == "native" for r in out["native_layout_routes"])
    print(
        path.name,
        len(out["pages"]),
        "pages,",
        len(out["markdown"]),
        "Markdown characters,",
        len(out["assets"]),
        "decoded assets; exact original host output passed",
        flush=True,
    )
assert not any(
    n == "rapid_doc" or n.startswith("rapid_doc.") or n == "pypdfium2" for n in sys.modules
)
print(
    "23 complete user PDF pages, real CPU formula/OCR, exact pages/Markdown/content_list/decoded assets; no full suite/CUDA or performance claims"
)
