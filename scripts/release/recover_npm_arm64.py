"""Finish the 0.1.1 npm release using verified binaries and an npm package alias."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from common import ROOT, checked_files, run, version
from npm_packages import root_package
from publish_npm import integrity, packages


def main():
    release = version()
    if release != "0.1.1":
        raise ValueError("This recovery is specific to the partially published 0.1.1 release")
    old = "rapid-doc-win32-arm64"
    scoped = "@hzkitty/" + old
    root = ROOT / "build/release/npm-root"
    root_package(root)
    run("npm", "pack", "--ignore-scripts", cwd=root)
    selected = packages()  # Validate the complete, original six-platform package set.
    source = next(path for name, path in selected if name == old)
    destination = ROOT / "build/release/npm-arm64-scoped"
    destination.mkdir(parents=True)
    with tarfile.open(source) as archive:
        archive.extractall(destination, filter="data")
    package = destination / "package"
    metadata = json.loads((package / "package.json").read_text())
    metadata["name"] = scoped
    (package / "package.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = json.loads((package / "runtime/manifest.json").read_text())
    checked_files(package / "runtime", manifest["files"])
    run("npm", "pack", "--ignore-scripts", cwd=package)
    replacement = next(package.glob("*.tgz"))
    metadata = json.loads((root / "package.json").read_text())
    metadata["optionalDependencies"][old] = f"npm:{scoped}@{release}"
    (root / "package.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for path in root.glob("*.tgz"):
        path.unlink()
    run("npm", "pack", "--ignore-scripts", cwd=root)
    selected = [(scoped, replacement) if name == old else (name, path) for name, path in selected]
    for name, path in selected:
        result = subprocess.run(["npm", "view", f"{name}@{release}", "dist.integrity", "--json"],
                                capture_output=True, text=True, cwd=ROOT)
        if result.returncode == 0:
            if json.loads(result.stdout) != integrity(path):
                raise ValueError(f"Existing package bytes differ: {name}@{release}")
            print("Already published, identical:", name, flush=True)
            continue
        if "E404" not in result.stdout + result.stderr:
            raise RuntimeError("Cannot check existing npm version: " + name)
        if name == "rapid-doc":
            # Exercise the real npm registry alias before publishing the loader.
            consumer = ROOT / "build/release/npm-alias-consumer"
            consumer.mkdir()
            (consumer / "package.json").write_text(json.dumps({"private": True, "dependencies": {
                old: f"npm:{scoped}@{release}"}}))
            run("npm", "install", "--force", "--ignore-scripts", "--no-audit", "--no-fund", cwd=consumer)
            shutil.copy2(root / "platform.js", consumer / "platform.js")
            run("node", "-e", "const fs=require('node:fs'); const {resolveAddon}=require('./platform'); "
                "const p=resolveAddon({}, 'win32', 'arm64'); if(!fs.existsSync(p))throw Error(p); console.log(p)",
                cwd=consumer, env={**os.environ, "RAPIDDOC_NODE_LIBRARY": ""})
            checked_files(consumer / "node_modules" / old / "runtime", manifest["files"])
        run("npm", "publish", path, "--access", "public", "--provenance", "--tag", "latest")


if __name__ == "__main__":
    main()
