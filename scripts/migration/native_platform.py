"""Platform names and shared-library conventions used by SDK packagers."""

import platform
from pathlib import Path
import subprocess
import sys


def host_platform(cuda=False):
    if sys.platform == "darwin":
        arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64"}[
            platform.machine().lower()
        ]
        if cuda:
            raise ValueError("macOS SDK packages use CPU execution")
        return "macos-" + arch
    return ("windows-x86_64" if sys.platform == "win32" else "linux-x86_64") + (
        "-cuda" if cuda else ""
    )


def shared_library(stem):
    if sys.platform == "win32":
        return stem + ".dll"
    return "lib" + stem + (".dylib" if sys.platform == "darwin" else ".so")


def is_shared_library(path):
    path = Path(path)
    return path.suffix in (".dll", ".dylib") or ".so" in path.name


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
