"""Prepare a Windows-only SxS research bundle; never edits a JDK or source ORT.

This prototype updates a private DLL's manifest resource, invalidating its
original Authenticode signature. It is not a signed release runtime package.
"""

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import shutil

# Reuse the production packager primitives; this CLI remains a research tool.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windows_ort_bundle import LIBRARIES, remove_private_manifest, text_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crt-directory", type=Path)
    args = parser.parse_args()
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("The prototype requires 64-bit Windows")
    source = args.ort.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = output / "RapidDoc.Runtime.PrivateCRT"
    directory.mkdir()
    crt = args.crt_directory or Path(os.environ["SystemRoot"]) / "System32"
    for name in LIBRARIES:
        shutil.copy2(crt / name, directory / name)
    target = directory / "onnxruntime.dll"
    shutil.copy2(source, target)
    remove_private_manifest(target)
    assert text_hash(source) == text_hash(target), "ORT execution code changed"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    identity = (
        'type="win32" name="RapidDoc.Runtime.PrivateCRT" '
        'version="1.0.0.0" processorArchitecture="amd64"'
    )
    assembly = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n'
    )
    files = "".join(f'  <file name="{name}" />\n' for name in (*LIBRARIES, "onnxruntime.dll"))
    (directory / "RapidDoc.Runtime.PrivateCRT.manifest").write_text(
        assembly + f"  <assemblyIdentity {identity} />\n" + files + "</assembly>\n", "utf8"
    )
    (output / "app.manifest").write_text(
        assembly
        + '<assemblyIdentity type="win32" name="RapidDoc.OrtLoader" version="1.0.0.0" />\n'
        + f"<dependency><dependentAssembly><assemblyIdentity {identity} /></dependentAssembly></dependency>\n"
        + "</assembly>\n",
        "utf8",
    )
    print("Research bundle:", output)


if __name__ == "__main__":
    main()
