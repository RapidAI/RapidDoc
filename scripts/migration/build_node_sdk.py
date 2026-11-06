"""Build an installable, platform-specific Node.js preview package."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from native_platform import shared_library, host_platform

ROOT = Path(__file__).resolve().parents[2]


def add_node_notices(directory, target):
    from generate_release_notice_policy import selected_files
    from shared_license_terms import pack_shared, unpack_shared

    text_file = directory / "licenses/sdk/THIRD_PARTY_NOTICES.txt"
    index_file = directory / "licenses/sdk/third-party-notices.json"
    entries = unpack_shared(text_file.read_bytes(), index_file.read_bytes())

    def graph(package):
        tree = subprocess.check_output(
            [
                "cargo",
                "tree",
                "--locked",
                "--offline",
                "-p",
                package,
                "--target",
                target,
                "--edges",
                "normal,no-proc-macro",
                "--prefix",
                "none",
                "--format",
                "{p}",
            ],
            cwd=ROOT,
            text=True,
        )
        return {
            (m[1], m[2]) for line in tree.splitlines() if (m := re.match(r"^(\S+) v(\S+)", line))
        }

    extra = graph("rapiddoc-node") - graph("rapiddoc-python")
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--locked", "--offline", "--format-version", "1"], cwd=ROOT
        )
    )
    node_licenses = ROOT / "bindings/node/licenses"
    external = {
        item["package"]: item for item in json.loads((node_licenses / "sources.json").read_text())
    }
    for package in metadata["packages"]:
        if (package["name"], package["version"]) not in extra or not package.get("source"):
            continue
        files, keep = selected_files(package)
        if not keep:
            name = package["name"] + "-" + package["version"]
            record = external.get(name)
            if record is None:
                raise ValueError("Missing Node dependency license: " + name)
            vcs = json.loads(
                (Path(package["manifest_path"]).parent / ".cargo_vcs_info.json").read_text()
            )
            data = (node_licenses / record["file"]).read_bytes()
            if (
                vcs["git"]["sha1"] != record["commit"]
                or hashlib.sha256(data).hexdigest() != record["sha256"]
            ):
                raise ValueError("Node license provenance mismatch: " + name)
            files, keep = {"LICENSE": data}, {"LICENSE"}
        for name in keep:
            entries[f"node/{package['name']}-{package['version']}/{name}"] = files[name]
    text, index = pack_shared(entries)
    text_file.write_bytes(text)
    index_file.write_bytes(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        required=True,
        help="Prepared shared runtime bundle (ORT, PDFium, GEOS, notices)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("Choose a fresh --output directory: " + str(output))
    runtime = args.runtime_dir.resolve(strict=True)
    manifest = json.loads((runtime / "manifest.json").read_text())
    # Reject partial or stale bundles before creating the npm package.
    for name, digest in manifest["files"].items():
        candidate = (runtime / name).resolve(strict=True)
        if not candidate.is_relative_to(runtime):
            raise SystemExit("Runtime manifest path escapes bundle")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise SystemExit("Runtime checksum mismatch: " + name)
    if not args.skip_build:
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [str(ROOT / ".venv/Scripts"), str(ROOT / "build/tools/nasm-3.01"), env["PATH"]]
        )
        subprocess.run(
            ["cargo", "build", "--release", "-p", "rapiddoc-node"], cwd=ROOT, env=env, check=True
        )
    windows = os.name == "nt"
    native = ROOT / "target/release" / shared_library("rapid_doc_node")
    output.mkdir(parents=True)
    source = ROOT / "bindings/node"
    for name in ("index.js", "index.d.ts", "worker.js", "progress.js", "platform.js", "README.md", "package.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    metadata = json.loads((output / "package.json").read_text())
    metadata["os"] = [sys.platform]
    metadata["cpu"] = ["arm64" if manifest["platform"] == "macos-aarch64" else "x64"]
    (output / "package.json").write_text(json.dumps(metadata, indent=2) + "\n")
    destination = output / "runtime"
    for name in manifest["files"]:
        # The shared bundle may also expose a C ABI. Node uses its own N-API addon.
        if name.startswith("include/") or Path(name).name in (
            "rapid_doc_c.dll",
            "librapid_doc_c.so",
            "librapid_doc_c.dylib",
        ):
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime / name, target)
    shutil.copy2(native, destination / "rapid_doc_node.node")
    target = {"macos-aarch64": "aarch64-apple-darwin", "macos-x86_64": "x86_64-apple-darwin"}.get(
        host_platform(), "x86_64-pc-windows-msvc" if windows else "x86_64-unknown-linux-gnu"
    )
    add_node_notices(destination, target)
    files = {
        p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file()
    }
    (destination / "manifest.json").write_text(
        json.dumps({"schema": 1, "platform": manifest["platform"], "files": files}, indent=2) + "\n"
    )
    subprocess.run(
        ["npm.cmd" if windows else "npm", "pack", "--ignore-scripts"], cwd=output, check=True
    )
    print("Node preview:", output)


if __name__ == "__main__":
    main()
