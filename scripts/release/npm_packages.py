"""Split verified standalone SDK archives into publishable npm platform packages."""

import argparse
import json
from pathlib import Path
import shutil

from common import ROOT, checked_files, run, version

PLATFORMS = {
    "linux-aarch64": ("rapid-doc-linux-arm64-gnu", "linux"),
    "windows-aarch64": ("@hzkitty/rapid-doc-win32-arm64", "win32"),
    "windows-x86_64": ("@hzkitty/rapid-doc-win32-x64", "win32"),
    "linux-x86_64": ("rapid-doc-linux-x64-gnu", "linux"),
    "macos-x86_64": ("rapid-doc-darwin-x64", "darwin"),
    "macos-aarch64": ("rapid-doc-darwin-arm64", "darwin"),
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
    (output / "README.md").write_text(
        f"# {name}\n\n"
        f"RapidDoc Node.js 的 `{platform}` 原生运行库平台包。\n\n"
        "通常由主包 `rapid-doc` 根据操作系统和 CPU 架构自动安装。\n"
        "应用通过主包调用，无需直接导入此平台包。\n\n"
        "```sh\nnpm install rapid-doc\n```\n\n"
        "```js\nconst { RapidDoc } = require('rapid-doc');\n```\n\n"
        "此包包含对应平台的原生引擎及运行库，不包含模型权重。\n\n"
        "[使用文档](https://github.com/RapidAI/RapidDoc/blob/main/bindings/node/README.md)"
        " · [项目源码](https://github.com/RapidAI/RapidDoc)\n",
        encoding="utf-8",
    )
    (output / "package.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": version(),
                "description": f"RapidDoc native runtime: {platform}",
                "license": "Apache-2.0",
                "os": [os_name],
                "cpu": ["arm64" if base.endswith("-aarch64") else "x64"],
                "repository": {
                    "type": "git",
                    "url": "git+https://github.com/RapidAI/RapidDoc.git",
                },
                "files": ["runtime", "LICENSE", "README.md"],
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
    # Keep the installed directory name compatible with the 0.1.1 loader.
    scoped_arm = "@hzkitty/rapid-doc-win32-arm64"
    metadata["optionalDependencies"].pop(scoped_arm)
    metadata["optionalDependencies"]["rapid-doc-win32-arm64"] = f"npm:{scoped_arm}@{version()}"
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
