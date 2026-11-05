"""Split verified standalone SDK archives into publishable npm platform packages."""

import argparse
import json
from pathlib import Path
import shutil

from common import ROOT, checked_files, run, version

PLATFORMS = {
    "windows-x86_64": ("rapid-doc-win32-x64", "win32"),
    "linux-x86_64": ("rapid-doc-linux-x64-gnu", "linux"),
}


def assemble(source, output):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "runtime/manifest.json").read_text())
    checked_files(source / "runtime", manifest["files"])
    platform = manifest["platform"]
    cuda = platform.endswith("-cuda")
    base = platform.removesuffix("-cuda")
    name, os_name = PLATFORMS[base]
    name += "-cuda" if cuda else ""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "runtime", output / "runtime")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    (output / "package.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": version(),
                "description": f"RapidDoc native runtime: {platform}",
                "license": "Apache-2.0",
                "os": [os_name],
                "cpu": ["x64"],
                "repository": {
                    "type": "git",
                    "url": "git+https://github.com/RapidAI/RapidDoc.git",
                },
                "files": ["runtime", "LICENSE"],
                "engines": {"node": ">=20"},
                **({"libc": ["glibc"]} if os_name == "linux" else {}),
            },
            indent=2,
        )
        + "\n"
    )
    return name


def root_package(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "bindings/node"
    metadata = json.loads((source / "package.json").read_text())
    metadata.pop("scripts", None)
    metadata["files"] = [name for name in metadata["files"] if name != "runtime"]
    metadata["optionalDependencies"] = {
        name: version() for name, _ in PLATFORMS.values()
    }
    metadata["repository"] = {
        "type": "git",
        "url": "git+https://github.com/RapidAI/RapidDoc.git",
    }
    for name in metadata["files"]:
        shutil.copy2(
            ROOT / "LICENSE" if name == "LICENSE" else source / name, output / name
        )
    (output / "package.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.source:
        assemble(args.source, args.output)
    else:
        root_package(args.output)
    run(
        "npm.cmd" if __import__("os").name == "nt" else "npm",
        "pack",
        "--ignore-scripts",
        cwd=args.output,
    )
