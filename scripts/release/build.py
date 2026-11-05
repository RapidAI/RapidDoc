"""Build CPU or CUDA preview artifacts on a clean Windows/Linux runner."""

import argparse
import json
import os
import shutil
import sys
import venv

from common import ROOT, OUT, check_version, run, sha
from runtime import stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--cuda", action="store_true")
    a = p.parse_args()
    check_version(a.version)
    if a.cuda and not os.environ.get("RAPIDDOC_OPERATOR_BUNDLE"):
        raise ValueError(
            "Set the platform CUDA_OPERATORS_* variable before requesting CUDA packaging"
        )
    platform, runtime, ort = stage(a.cuda)
    output = OUT / platform
    run("cargo", "fetch", "--locked")
    for script in [
        "generate_release_notice_policy.py",
        "sync_migration_notices.py",
        "compact_release_notices.py",
    ]:
        run(sys.executable, ROOT / "scripts/migration" / script)
    run(
        "cargo",
        "build",
        "--release",
        "--locked",
        "-p",
        "rapiddoc-python",
        "-p",
        "rapiddoc-java",
        "-p",
        "rapiddoc-c",
        "-p",
        "rapiddoc-node",
        "--features",
        "rapiddoc-python/cuda,pyo3/extension-module",
    )
    migration = ROOT / "scripts/migration"
    native = output / "native"
    run(
        sys.executable,
        migration / "build_go_sdk.py",
        "--skip-build",
        "--ort-dir",
        ort,
        "--runtime-dir",
        runtime,
        "--output",
        native,
        *(["--cuda"] if a.cuda else []),
    )
    standalone = output / "node-standalone"
    run(
        sys.executable,
        migration / "build_node_sdk.py",
        "--skip-build",
        "--runtime-dir",
        native,
        "--output",
        standalone,
    )
    run(
        sys.executable,
        ROOT / "scripts/release/npm_packages.py",
        "--source",
        standalone,
        "--output",
        output / "npm",
    )
    run(
        sys.executable,
        migration / "build_java_sdk.py",
        "--skip-build",
        "--skip-tests",
        "--ort-dir",
        ort,
        "--runtime-dir",
        runtime,
        *(["--cuda"] if a.cuda else []),
    )
    mvn = shutil.which("mvn.cmd" if os.name == "nt" else "mvn")
    run(
        mvn,
        "-B",
        "-f",
        "bindings/java/pom.xml",
        "-Prelease",
        "-DskipTests",
        "-Drapid-doc.native.classifier=" + platform,
        "package",
    )
    java = output / "java"
    java.mkdir()
    for directory in [ROOT / "build/java-sdk", ROOT / "bindings/java/target"]:
        for f in directory.iterdir():
            if f.suffix in [".jar", ".pom"]:
                shutil.copy2(f, java / f.name)
    shutil.copy2(ROOT / "bindings/java/pom.xml", java / f"rapid-doc-{a.version}.pom")
    shutil.copy2(
        ROOT / "bindings/java/native/pom.xml",
        java / f"rapid-doc-native-{a.version}.pom",
    )
    raw = output / "wheel-raw"
    run(
        "maturin",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        "bindings/python/Cargo.toml",
        "--features",
        "cuda",
        "--out",
        raw,
        *(
            ["--compatibility", "linux", "--auditwheel", "skip"]
            if os.name != "nt"
            else []
        ),
    )
    wheels = output / "python"
    wheels.mkdir()
    for f in raw.glob("*.whl"):
        if os.name == "nt":
            shutil.copy2(f, wheels / f.name)
        else:
            run(
                "auditwheel",
                "repair",
                "--plat",
                "manylinux_2_34_x86_64",
                "--exclude",
                "libcudart.so.12",
                "--exclude",
                "libcudnn.so.9",
                "--wheel-dir",
                wheels,
                f,
            )
    run(sys.executable, "-m", "twine", "check", *wheels.glob("*.whl"))
    run(
        "go",
        "test",
        "./...",
        cwd=ROOT / "bindings/go",
        env={**os.environ, "RAPIDDOC_TEST_ROOT": ""},
    )
    run(
        mvn,
        "-B",
        "-f",
        "bindings/java/pom.xml",
        "-Dtest=InputOverloadsTest",
        "-Drapid-doc.native.classifier=" + platform,
        "test",
    )
    smoke(output, native, standalone, mvn)
    # The original node archive remains a convenient standalone download.
    shutil.copy2(
        native.with_suffix(".zip"),
        output / f"rapiddoc-native-{a.version}-{platform}.zip",
    )
    delivered = [
        p
        for directory in ["python", "npm", "java", "reports"]
        for p in (output / directory).iterdir()
        if p.is_file()
    ]
    delivered += list(output.glob("rapiddoc-native-*.zip"))
    (output / "checksums.json").write_text(
        json.dumps(
            {p.relative_to(output).as_posix(): sha(p) for p in delivered}, indent=2
        )
    )


def smoke(output, native, standalone, mvn):
    os.environ["RAPIDDOC_TEST_ROOT"] = str(ROOT)
    os.environ["RAPIDDOC_RUNTIME_DIR"] = str(native)
    # Hosted builds validate CPU execution even when packaging GPU libraries.
    # Actual CUDA inference belongs on the explicit GPU acceptance runner.
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    reports = output / "reports"
    reports.mkdir()
    consumer = output / "consumer"
    consumer.mkdir()
    venv.create(consumer / "python", with_pip=True)
    python = (
        consumer
        / "python"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    run(python, "-m", "pip", "install", *sorted((output / "python").glob("*.whl")))
    os.environ["RAPIDDOC_SMOKE_REPORT"] = str(reports / "python.json")
    run(python, "-I", ROOT / "scripts/release/smoke_python.py", cwd=consumer)
    run(python, "-I", ROOT / "bindings/python/tests/lifecycle_install.py", ROOT, cwd=consumer)
    os.environ["RAPIDDOC_NATIVE_LIBRARY"] = str(
        native / ("rapid_doc_c.dll" if os.name == "nt" else "librapid_doc_c.so")
    )
    os.environ["RAPIDDOC_SMOKE_REPORT"] = str(reports / "go.json")
    run(
        "go", "run", ROOT / "scripts/release/smoke_go/main.go", cwd=ROOT / "bindings/go"
    )
    npm = "npm.cmd" if os.name == "nt" else "npm"
    node = consumer / "node"
    node.mkdir()
    run(
        npm,
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        *standalone.glob("*.tgz"),
        cwd=node,
    )
    shutil.copy2(ROOT / "scripts/release/smoke_node.cjs", node / "smoke.cjs")
    os.environ["RAPIDDOC_SMOKE_REPORT"] = str(reports / "node.json")
    run("node", "smoke.cjs", cwd=node)
    classpath_file = output / "classpath.txt"
    run(
        mvn,
        "-B",
        "-f",
        "bindings/java/pom.xml",
        "dependency:build-classpath",
        "-Drapid-doc.native.classifier=" + output.name,
        f"-Dmdep.outputFile={classpath_file}",
    )
    classpath = (
        str(ROOT / "bindings/java/target/classes")
        + os.pathsep
        + classpath_file.read_text().strip()
    )
    run(
        "javac",
        "--release",
        "8",
        "-cp",
        classpath,
        "-d",
        consumer,
        ROOT / "scripts/release/Smoke.java",
    )
    os.environ["RAPIDDOC_SMOKE_REPORT"] = str(reports / "java.json")
    run(
        "java",
        "-Xcheck:jni",
        "-cp",
        str(consumer) + os.pathsep + classpath,
        "io.github.rapidai.rapiddoc.Smoke",
    )
    expected = json.loads((reports / "python.json").read_text())
    for language in ["go", "node", "java"]:
        if json.loads((reports / f"{language}.json").read_text()) != expected:
            raise ValueError(
                f"Same-platform seven-format binding output differs: {language}; see {reports}"
            )
    print("Seven-format installation and binding equivalence: PASS")


if __name__ == "__main__":
    main()
