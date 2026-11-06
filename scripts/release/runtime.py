"""Stage runtime assets from pinned distributions into a clean CI checkout."""

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
import zipfile

from common import ROOT, OUT, checked_files, run, sha, host_platform, shared_library

MIGRATION = ROOT / "scripts/migration"
sys.path.insert(0, str(MIGRATION))
from license_bundle import compact
from native_platform import is_shared_library, relocate_dylibs

ORT_VERSION = "1.23.2"
ORT = {
    "linux-aarch64": (
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
    ),
    "windows-aarch64": (
        "onnxruntime-win-arm64-1.23.2.zip",
        "1cfe88b6435df3b5fb0e9f6bd7d6f5df1e887b6174de7f6e2a47bab956f3f168",
    ),
    "macos-aarch64": (
        "onnxruntime-osx-arm64-1.23.2.tgz",
        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
    ),
    "macos-x86_64": (
        "onnxruntime-osx-x86_64-1.23.2.tgz",
        "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6",
    ),
    "windows-x86_64": (
        "onnxruntime-win-x64-1.23.2.zip",
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
    ),
    "windows-x86_64-cuda": (
        "onnxruntime-win-x64-gpu-1.23.2.zip",
        "e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9",
    ),
    "linux-x86_64": (
        "onnxruntime-linux-x64-1.23.2.tgz",
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
    ),
    "linux-x86_64-cuda": (
        "onnxruntime-linux-x64-gpu-1.23.2.tgz",
        "2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53",
    ),
}


def download(url, destination, digest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        with (
            urllib.request.urlopen(url, timeout=120) as source,
            temporary.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        if sha(temporary) != digest:
            temporary.unlink()
            raise ValueError("Downloaded archive checksum differs: " + url)
        temporary.replace(destination)
    if sha(destination) != digest:
        raise ValueError("Cached archive checksum differs: " + str(destination))


def build_windows_arm_geos(archive, destination, output):
    """Shapely 2.1.2 has no Windows ARM64 wheel; build its pinned GEOS instead."""
    source = output / "geos-source"
    source.mkdir(parents=True)
    with tarfile.open(archive) as package:
        package.extractall(source, filter="data")
    source = source / "geos-3.13.1"
    build = output / "geos-build"
    install = output / "geos-install"
    run("cmake", "-S", source, "-B", build, "-G", "Ninja",
        "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl",
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTING=OFF", "-DBUILD_BENCHMARKS=OFF",
        "-DBUILD_DOCUMENTATION=OFF", f"-DCMAKE_INSTALL_PREFIX={install}")
    run("cmake", "--build", build, "--parallel", "2")
    run("cmake", "--install", build)
    for library in (install / "bin").glob("*.dll"):
        shutil.copy2(library, destination / library.name)
    if not (destination / "geos_c.dll").is_file():
        raise FileNotFoundError("GEOS ARM64 build did not install geos_c.dll")
    notices = destination / "licenses"
    notices.mkdir()
    for name in ("COPYING", "COPYING.LESSER", "AUTHORS"):
        file = source / name
        if file.is_file():
            shutil.copy2(file, notices / name)
            shutil.copy2(file, destination / ("LICENSE_" + name))
    if not (notices / "COPYING").is_file():
        raise FileNotFoundError("GEOS source license is missing")


def extract_ort(archive, destination):
    """Extract real libraries and notices; loaders accept versioned ORT names.

    The upstream Unix aliases are symlinks, not additional runtime binaries.
    Keeping only regular members avoids expanding them into duplicate payloads.
    """
    def selected(name):
        parts = Path(name).parts
        base = Path(name).name
        return (
            len(parts) == 3 and parts[-2] == "lib" and is_shared_library(base)
        ) or (len(parts) == 2 and base in ["LICENSE", "ThirdPartyNotices.txt"])

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as package:
            for entry in package.infolist():
                if not entry.is_dir() and selected(entry.filename):
                    (destination / Path(entry.filename).name).write_bytes(package.read(entry))
    else:
        with tarfile.open(archive) as package:
            for entry in package:
                if entry.isfile() and selected(entry.name):
                    (destination / Path(entry.name).name).write_bytes(package.extractfile(entry).read())


def stage(cuda=False):
    windows = os.name == "nt"
    platform = host_platform(cuda)
    runtime = ROOT / "bindings/python/python/rapid_doc_rust/runtime"
    if runtime.exists():
        raise ValueError(
            "Use a clean checkout: runtime staging directory already exists"
        )
    runtime.mkdir(parents=True)
    raw = Path(metadata.distribution("pypdfium2").locate_file("pypdfium2_raw"))
    library = shared_library("pdfium")
    target = shared_library("rapiddoc_pdfium")
    shutil.copy2(raw / library, runtime / target)
    licenses = raw / "licenses"
    if not licenses.is_dir():
        distribution_root = Path(metadata.distribution("pypdfium2").locate_file("."))
        candidates = sorted(distribution_root.glob("pypdfium2-*.dist-info/licenses"))
        if candidates:
            licenses = candidates[0]
    if not licenses.is_dir():
        raise FileNotFoundError("pypdfium2 license directory is missing")
    shutil.copytree(licenses, runtime / "licenses")
    compact(runtime / "licenses")
    (runtime / "manifest.json").write_text(
        json.dumps(
            {
                "platform": sys.platform,
                "library": target,
                "sha256": sha(runtime / target),
                "version": json.loads((raw / "version.json").read_text()),
                "source": "pypdfium2==5.10.1",
                "license_files": {
                    p.relative_to(runtime).as_posix(): sha(p)
                    for p in (runtime / "licenses").rglob("*")
                    if p.is_file()
                },
            },
            indent=2,
        )
    )
    dest = runtime / "geos"
    dest.mkdir()
    source = OUT / "downloads/geos-3.13.1.tar.bz2"
    download(
        "https://download.osgeo.org/geos/geos-3.13.1.tar.bz2", source,
        "df2c50503295f325e7c8d7b783aca8ba4773919cde984193850cf9e361dfd28c",
    )
    if platform == "windows-aarch64":
        build_windows_arm_geos(source, dest, OUT / platform)
    else:
        distribution = metadata.distribution("shapely")
        geos = Path(distribution.locate_file("shapely/.dylibs" if sys.platform == "darwin" else "shapely.libs"))
        notices = Path(distribution._path) / "licenses"
        for p in geos.iterdir():
            if p.is_file() and is_shared_library(p):
                shutil.copy2(p, dest / p.name)
        if not list(dest.glob("*geos_c*")):
            raise ValueError("Paired Shapely GEOS distribution missing")
        shutil.copytree(notices, dest / "licenses")
        for i, p in enumerate(sorted(notices.rglob("*"))):
            if p.is_file():
                shutil.copy2(p, dest / f"LICENSE_{i}_{p.name}")
    shutil.copy2(source, dest / source.name)
    if sys.platform == "darwin":
        relocate_dylibs(dest)
    elif not windows:
        for p in dest.glob("libgeos_c*.so*"):
            run("patchelf", "--set-rpath", "$ORIGIN", p)
    archive_name, digest = ORT[platform]
    archive = OUT / "downloads" / archive_name
    download(
        f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{archive_name}",
        archive,
        digest,
    )
    ort = OUT / platform / "ort"
    ort.mkdir(parents=True)

    extract_ort(archive, ort)
    if sys.platform == "darwin":
        relocate_dylibs(ort)
    if cuda:
        operators = Path(os.environ["RAPIDDOC_OPERATOR_BUNDLE"])
        manifest = json.loads((operators / "manifest.json").read_text())
        if manifest.get("platform") != platform:
            raise ValueError("CUDA operator bundle platform mismatch")
        checked_files(operators, manifest["files"])
        for name in manifest["files"]:
            output = runtime / "onnx_ops" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(operators / name, output)
        expected = "onnxruntime_ops.dll" if windows else "libonnxruntime_ops.so"
        if (
            not (runtime / "onnx_ops" / expected).is_file()
            or not (runtime / "onnx_ops/licenses").is_dir()
        ):
            raise ValueError("CUDA operators and their licenses are required")
    return platform, runtime, ort


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    print(stage(p.parse_args().cuda))
