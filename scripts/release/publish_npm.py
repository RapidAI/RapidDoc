"""Publish exact verified tarballs, resuming a partially successful release."""

import base64
import hashlib
import json
import subprocess
import tarfile

from common import ROOT, check_version, run, version


def integrity(path):
    return (
        "sha512-"
        + base64.b64encode(hashlib.sha512(path.read_bytes()).digest()).decode()
    )


def packages():
    paths = sorted((ROOT / "artifacts").glob("sdk-*/npm/*.tgz"))
    paths += list((ROOT / "build/release/npm-root").glob("*.tgz"))
    found = {}
    allowed = {
        "rapid-doc",
        "rapid-doc-win32-x64",
        "rapid-doc-linux-x64-gnu",
        "rapid-doc-win32-x64-cuda",
        "rapid-doc-linux-x64-gnu-cuda",
    }
    for p in paths:
        with tarfile.open(p) as archive:
            manifest = json.load(archive.extractfile("package/package.json"))
        name = manifest["name"]
        if name not in allowed or name in found or manifest["version"] != version():
            raise ValueError("Unexpected, duplicate or mismatched npm package: " + name)
        found[name] = p
    if (
        not {"rapid-doc", "rapid-doc-win32-x64", "rapid-doc-linux-x64-gnu"}
        <= found.keys()
    ):
        raise ValueError("The loader and both CPU platform packages must be present")
    return [(name, p) for name, p in found.items() if name != "rapid-doc"] + [
        ("rapid-doc", found["rapid-doc"])
    ]


def main():
    check_version(version(), publishing=True)
    for name, path in packages():
        result = subprocess.run(
            ["npm", "view", f"{name}@{version()}", "dist.integrity", "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if json.loads(result.stdout) != integrity(path):
                raise ValueError(
                    f"{name}@{version()} already exists with different bytes; never overwrite it"
                )
            print("Already published, identical:", name)
            continue
        if "E404" not in result.stdout + result.stderr:
            raise RuntimeError("Cannot check existing npm version: " + name)
        run(
            "npm",
            "publish",
            path,
            "--access",
            "public",
            "--provenance",
            "--tag",
            "preview",
        )


if __name__ == "__main__":
    main()
