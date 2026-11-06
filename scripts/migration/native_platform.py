"""Platform names and shared-library conventions used by SDK packagers."""

import platform
import hashlib
import re
from pathlib import Path
import subprocess
import sys


def host_architecture():
    return {"amd64": "x86_64", "x86_64": "x86_64",
            "arm64": "aarch64", "aarch64": "aarch64"}[platform.machine().lower()]


def host_platform(cuda=False):
    arch = host_architecture()
    system = {"win32": "windows", "linux": "linux", "darwin": "macos"}[sys.platform]
    if cuda and (system == "macos" or arch == "aarch64"):
        raise ValueError("macOS and ARM64 SDK packages use CPU execution")
    return system + "-" + arch + ("-cuda" if cuda else "")


def rust_target():
    arch = host_architecture()
    return arch + {"win32": "-pc-windows-msvc", "linux": "-unknown-linux-gnu",
                   "darwin": "-apple-darwin"}[sys.platform]


def shared_library(stem):
    if sys.platform == "win32":
        return stem + ".dll"
    return "lib" + stem + (".dylib" if sys.platform == "darwin" else ".so")


def is_shared_library(path):
    path = Path(path)
    return path.suffix in (".dll", ".dylib") or ".so" in path.name


def ort_library_files(directory):
    """Keep the real versioned ORT library and distinct provider libraries.

    Older staging directories may contain expanded symlinks. Only omit their
    aliases when the bytes match, leaving custom or differently signed files intact.
    """
    files = sorted(p for p in Path(directory).iterdir() if p.is_file() and is_shared_library(p))
    versioned = [p for p in files if re.fullmatch(
        r"libonnxruntime(?:\.so(?:\.\d+)+|(?:\.\d+)+\.dylib)", p.name
    )]
    aliases = {"libonnxruntime.so", "libonnxruntime.so.1", "libonnxruntime.dylib"}
    # .so.1 can itself be the only real SONAME library in an external bundle.
    real = [p for p in versioned if p.name != "libonnxruntime.so.1"]
    if len(real) != 1 or not any(p.name in aliases for p in files):
        return files

    def digest(path):
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.digest()

    expected = digest(real[0])
    return [p for p in files if p.name not in aliases or digest(p) != expected]


def relocate_dylibs(directory):
    """Make copied macOS libraries resolve bundled siblings after extraction."""
    files = {p.name: p for p in Path(directory).glob("*.dylib")}
    for name, path in files.items():
        dependencies = subprocess.check_output(["otool", "-L", str(path)], text=True)
        command = ["install_name_tool", "-id", "@rpath/" + name]
        for line in dependencies.splitlines()[2:]:
            dependency = line.strip().split(" (", 1)[0]
            sibling = Path(dependency).name
            if sibling in files:
                command += ["-change", dependency, "@loader_path/" + sibling]
        subprocess.run(command + [str(path)], check=True)
        subprocess.run(["codesign", "--force", "--sign", "-", str(path)], check=True)
