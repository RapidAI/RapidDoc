"""Format owned Rust, Python bindings, Java and hand-written TOML/XML config.

Run with --check for a read-only check. Requires rustfmt, Ruff, Taplo and
JDK 17+ for google-java-format; this does not change Java's release target.
"""

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
from xml.dom import minidom

ROOT = Path(__file__).resolve().parents[1]
JAVA_FORMAT_VERSION = "1.24.0"
JAVA_FORMAT_SHA256 = "812f805f58112460edf01bf202a8e61d0fd1f35c0d4fabd54220640776ec57a1"
TAPLO_SHA256 = "12660d24becf05a78b39133112e91d1bd319b22b8e49660448c222a6858a6827"


def owned(path):
    generated = {
        ROOT / "rust/rapiddoc-office/src/cfb_uppercase.rs",
        ROOT / "rust/rapiddoc-office/src/docx/decimal_data.rs",
    }
    return path not in generated and not {
        "vendor",
        "target",
        "runtime",
        "__pycache__",
    }.intersection(path.parts)


def files():
    rust = list((ROOT / "rust").rglob("*.rs"))
    rust += list((ROOT / "bindings").glob("*/src/*.rs"))
    rust += list((ROOT / "scripts/migration/python_native").glob("*.rs"))
    python = list((ROOT / "bindings/python").rglob("*.py")) + [Path(__file__)]
    java = list((ROOT / "bindings/java/src").rglob("*.java"))
    toml = [
        ROOT / name for name in ("Cargo.toml", "pyproject.toml", ".rustfmt.toml", ".taplo.toml")
    ]
    toml += list((ROOT / "rust").rglob("Cargo.toml"))
    toml += list((ROOT / "bindings").glob("*/Cargo.toml"))
    toml += [ROOT / "bindings/python/pyproject.toml", ROOT / "bindings/python/ruff.toml"]
    xml = [ROOT / "bindings/java/pom.xml", ROOT / "bindings/java/native/pom.xml"]
    return {
        kind: sorted({p.resolve() for p in paths if owned(p)})
        for kind, paths in dict(rust=rust, python=python, java=java, toml=toml, xml=xml).items()
    }


def format_xml(path):
    document = minidom.parseString(path.read_bytes())

    def remove_indent(node):
        for child in list(node.childNodes):
            if child.nodeType == child.TEXT_NODE and not child.data.strip():
                node.removeChild(child)
                child.unlink()
            else:
                remove_indent(child)

    remove_indent(document)
    return document.toprettyxml(indent="  ", encoding="utf-8")


def java_formatter():
    name = f"google-java-format-{JAVA_FORMAT_VERSION}-all-deps.jar"
    path = ROOT / "build/tools" / name
    if not path.is_file():
        url = (
            "https://repo.maven.apache.org/maven2/com/google/googlejavaformat/"
            f"google-java-format/{JAVA_FORMAT_VERSION}/{name}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != JAVA_FORMAT_SHA256:
            raise RuntimeError("Java formatter SHA-256 mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != JAVA_FORMAT_SHA256:
        raise RuntimeError("Java formatter SHA-256 mismatch")
    return path


def taplo_formatter():
    if os.name != "nt":
        executable = shutil.which("taplo")
        if not executable:
            raise RuntimeError("Install Taplo 0.10.0 to format TOML")
        return executable
    path = ROOT / "build/tools/taplo-0.10.0.exe"
    if not path.is_file():
        url = "https://github.com/tamasfe/taplo/releases/download/0.10.0/taplo-windows-x86_64.gz"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = gzip.decompress(response.read())
        if hashlib.sha256(data).hexdigest() != TAPLO_SHA256:
            raise RuntimeError("Taplo SHA-256 mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TAPLO_SHA256:
        raise RuntimeError("Taplo SHA-256 mismatch")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--java", default="java", help="JDK 17+ java executable for formatting")
    args = parser.parse_args()
    selected = files()
    paths = lambda kind: [str(p.relative_to(ROOT)) for p in selected[kind]]

    def run(command):
        subprocess.run([str(v) for v in command], cwd=ROOT, check=True)

    rust = paths("rust")
    for start in range(0, len(rust), 50):
        run(
            [
                "rustfmt",
                "--config-path",
                ".rustfmt.toml",
                "--config",
                "skip_children=true",
                *(["--check"] if args.check else []),
                *rust[start : start + 50],
            ]
        )
    run(
        [
            "ruff",
            "format",
            "--config",
            "bindings/python/ruff.toml",
            "--no-cache",
            *(["--check"] if args.check else []),
            *paths("python"),
        ]
    )
    run(
        [
            args.java,
            "-jar",
            java_formatter(),
            "--aosp",
            "--skip-sorting-imports",
            "--skip-removing-unused-imports",
            "--skip-reflowing-long-strings",
            "--skip-javadoc-formatting",
            *(["--dry-run", "--set-exit-if-changed"] if args.check else ["--replace"]),
            *paths("java"),
        ]
    )
    run(
        [
            taplo_formatter(),
            "fmt",
            *(["--check"] if args.check else []),
            *paths("toml"),
        ]
    )
    for path in selected["xml"]:
        formatted = format_xml(path)
        if args.check:
            if path.read_bytes() != formatted:
                raise RuntimeError(f"XML formatting differs: {path.relative_to(ROOT)}")
        else:
            path.write_bytes(formatted)
    print(
        "Checked" if args.check else "Formatted",
        {kind: len(value) for kind, value in selected.items()},
    )


if __name__ == "__main__":
    main()
