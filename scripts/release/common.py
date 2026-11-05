"""Shared SDK release checks; no publishing side effects on import."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/release"
NS = {"m": "http://maven.apache.org/POM/4.0.0"}


def run(*args, **kwargs):
    return subprocess.run(
        [str(a) for a in args], check=True, cwd=kwargs.pop("cwd", ROOT), **kwargs
    )


def version():
    return tomllib.loads((ROOT / "Cargo.toml").read_text("utf8"))["workspace"][
        "package"
    ]["version"]


def check_version(value, publishing=False):
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
        raise ValueError("Use an SDK release version such as 0.1.0")
    versions = [
        version(),
        json.loads((ROOT / "bindings/node/package.json").read_text())["version"],
        tomllib.loads((ROOT / "bindings/python/pyproject.toml").read_text())["project"][
            "version"
        ],
    ]
    for name in ["pom.xml", "native/pom.xml"]:
        versions.append(
            ET.parse(ROOT / "bindings/java" / name)
            .getroot()
            .findtext("m:version", namespaces=NS)
        )
    if any(v != value for v in versions):
        raise ValueError(
            f"SDK versions differ: requested={value}, manifests={versions}"
        )
    if publishing and os.environ.get("GITHUB_REF") != f"refs/tags/sdk-v{value}":
        raise ValueError(
            "Publishing requires running the workflow on the matching sdk-vVERSION tag"
        )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_files(directory, files):
    directory = Path(directory).resolve()
    for name, digest in files.items():
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or sha(path) != digest
        ):
            raise ValueError(f"Missing, escaping or changed release payload: {name}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("version")
    p.add_argument("--publishing", action="store_true")
    a = p.parse_args()
    check_version(a.version, a.publishing)
    print("SDK version verified:", a.version)
