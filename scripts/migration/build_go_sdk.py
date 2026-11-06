"""Build a self-contained native runtime directory for the source-preview Go SDK."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from native_platform import host_platform, shared_library, is_shared_library

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-dir", type=Path, default=ROOT / "target/release/runtime")
    parser.add_argument(
        "--runtime-dir", type=Path, default=ROOT / "bindings/python/python/rapid_doc_rust/runtime"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--external-ort", action="store_true")
    args = parser.parse_args()
    windows = os.name == "nt"
    platform = host_platform(args.cuda)
    output = (args.output or ROOT / "build/go-sdk" / platform).resolve()
    if output.exists():
        raise SystemExit("Choose a fresh --output directory: " + str(output))
    if not args.skip_build:
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [str(ROOT / ".venv/Scripts"), str(ROOT / "build/tools/nasm-3.01"), env["PATH"]]
        )
        subprocess.run(
            ["cargo", "build", "--release", "-p", "rapiddoc-c"], cwd=ROOT, env=env, check=True
        )
    name = shared_library("rapid_doc_c")
    files = {name: (ROOT / "target/release" / name).read_bytes()}
    ort = args.ort_dir.resolve(strict=True)
    shared = is_shared_library
    gpu = any("providers_cuda" in p.name for p in ort.iterdir())
    if gpu != args.cuda:
        raise SystemExit("ORT provider payload differs from --cuda selection")
    if windows and not args.external_ort:
        from windows_ort_bundle import bundle_files

        files.update(bundle_files(ort))
    else:
        for path in ort.iterdir():
            if path.is_file() and shared(path):
                files[path.name] = path.read_bytes()
    for notice in ("LICENSE", "ThirdPartyNotices.txt"):
        files["licenses/ort/" + notice] = (ort / notice).read_bytes()
    runtime = args.runtime_dir.resolve(strict=True)
    pdfium = shared_library("rapiddoc_pdfium")
    files[pdfium] = (runtime / pdfium).read_bytes()
    for path in (runtime / "geos").iterdir():
        if path.is_file() and (shared(path) or path.name.startswith("LICENSE") or path.name.endswith(".tar.bz2")):
            files["geos/" + path.name] = path.read_bytes()
    for source, prefix in [
        (runtime / "licenses", "licenses/pdfium"),
        (ROOT / "bindings/python/python/rapid_doc_rust/licenses", "licenses/sdk"),
    ]:
        for path in source.rglob("*"):
            if path.is_file():
                files[prefix + "/" + path.relative_to(source).as_posix()] = path.read_bytes()
    if args.cuda:
        operators = runtime / "onnx_ops"
        required = "onnxruntime_ops.dll" if windows else "libonnxruntime_ops.so"
        if not (operators / required).is_file():
            raise SystemExit("CUDA operator DLL missing")
        for path in operators.rglob("*"):
            if path.is_file() and (shared(path) or "licenses" in path.parts):
                files["onnx_ops/" + path.relative_to(operators).as_posix()] = path.read_bytes()
    files["include/rapid_doc.h"] = (ROOT / "bindings/c/include/rapid_doc.h").read_bytes()
    manifest = dict(
        schema=1,
        abi=1,
        platform=platform,
        files={name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    )
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.mkdir(parents=True)
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for name, data in files.items():
            package.writestr(name, data)
    print("Go native preview:", output / name)
    print("Deployment archive:", archive)


if __name__ == "__main__":
    main()
