"""Build/install the Java SDK and its platform runtime into local Maven only."""

import argparse
import ctypes
import re
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from native_platform import host_platform, shared_library, is_shared_library

ROOT = Path(__file__).resolve().parents[2]


def write_bundle(output, artifact, version, classifier, platform, files, namespace, **metadata):
    manifest = dict(
        schema=1,
        platform=platform,
        **metadata,
        files={name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    )
    jar = output / f"{artifact}-{version}-{classifier}.jar"
    with zipfile.ZipFile(jar, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(f"META-INF/{namespace}/{platform}/{name}", data)
        archive.writestr(
            f"META-INF/{namespace}/{platform}/manifest.json",
            (json.dumps(manifest, indent=2) + "\n").encode(),
        )
    return jar


def ort_version(path):
    library = ctypes.WinDLL(str(path)) if os.name == "nt" else ctypes.CDLL(str(path))
    library.OrtGetApiBase.restype = ctypes.POINTER(ctypes.c_void_p)
    call = ctypes.WINFUNCTYPE if os.name == "nt" else ctypes.CFUNCTYPE
    return call(ctypes.c_char_p)(library.OrtGetApiBase()[1])().decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ort-dir", type=Path, help="Directory containing the matching ORT shared libraries"
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=ROOT / "bindings/python/python/rapid_doc_rust/runtime",
        help="Prepared RapidDoc PDFium/GEOS runtime bundle",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--ort-version", default="1.23.2", help="Independent rapid-doc-ort artifact version"
    )
    parser.add_argument(
        "--ort-only", action="store_true", help="Package/install only ORT; no Rust or Java rebuild"
    )
    parser.add_argument(
        "--crt-dir",
        type=Path,
        help="Windows x64 Visual Studio CRT 14.51 redist directory; otherwise auto-discovered",
    )
    parser.add_argument(
        "--external-ort",
        action="store_true",
        help="Package unmodified external ORT without private CRT isolation",
    )
    parser.add_argument(
        "--ort-sign-command-json",
        help="Optional publisher signing argv as JSON; use {dll} for the derived ORT path",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--maven-settings", type=Path, help="Optional Maven settings; does not modify user settings"
    )
    args = parser.parse_args()
    match = re.fullmatch(r"1\.(\d+)\.(\d+)", args.ort_version)
    if not match or int(match[1]) < 23:
        raise SystemExit("ORT version must be a release version >=1.23.0")
    windows = os.name == "nt"
    platform = host_platform()
    classifier = platform + ("-cuda" if args.cuda else "")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(ROOT / ".venv/Scripts"), str(ROOT / "build/tools/nasm-3.01"), env["PATH"]]
    )

    def run(command):
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if not args.skip_build and not args.ort_only:
        command = ["cargo", "build", "--release", "--locked", "-p", "rapiddoc-java"]
        if args.cuda:
            command += ["--features", "cuda"]
        run(command)
    runtime = args.runtime_dir.resolve()
    ort = args.ort_dir
    if ort is None:
        # Development convenience only; this path is not written to the JAR.
        candidate = ROOT / "target/release/runtime"
        if candidate.is_dir():
            ort = candidate
        else:
            raise SystemExit("Supply --ort-dir with CPU/GPU ORT shared libraries")
    ort = ort.resolve()
    shared = is_shared_library
    ort_files = [p for p in ort.iterdir() if p.is_file() and shared(p)]
    if not any(p.name.startswith(("onnxruntime.", "libonnxruntime.")) for p in ort_files):
        raise SystemExit("ORT main library missing")
    if args.cuda and not any("providers_cuda" in p.name for p in ort_files):
        raise SystemExit("CUDA build requires GPU ORT, not CPU ORT")
    if not args.cuda and any("providers_cuda" in p.name for p in ort_files):
        raise SystemExit("GPU ORT requires --cuda; it cannot be labeled as a CPU runtime artifact")
    jni = shared_library("rapid_doc_jni")
    actual_version = ort_version(
        next(p for p in ort_files if p.name.startswith(("onnxruntime.", "libonnxruntime.")))
    )
    if actual_version != args.ort_version:
        raise SystemExit(
            f"ORT library version {actual_version} differs from --ort-version {args.ort_version}"
        )
    ort_payload = {}
    if windows and not args.external_ort:
        from windows_ort_bundle import bundle_files

        signer = None
        if args.ort_sign_command_json:
            signing = json.loads(args.ort_sign_command_json)
            if (
                not isinstance(signing, list)
                or not signing
                or not all(isinstance(v, str) for v in signing)
                or not any("{dll}" in v for v in signing)
            ):
                raise SystemExit(
                    "Signing command must be a nonempty JSON argv array containing {dll}"
                )
            signer = lambda path: run([v.replace("{dll}", str(path)) for v in signing])
        ort_payload.update(bundle_files(ort, args.crt_dir, signer))
    else:
        for p in ort_files:
            ort_payload[p.name] = p.read_bytes()
    for name in ("LICENSE", "ThirdPartyNotices.txt"):
        if not (ort / name).is_file():
            raise SystemExit("ORT attribution missing: " + str(ort / name))
        ort_payload["licenses/ort/" + name] = (ort / name).read_bytes()
    if args.cuda:
        operators = runtime / "onnx_ops"
        expected_ops = "onnxruntime_ops.dll" if windows else "libonnxruntime_ops.so"
        if not (operators / expected_ops).is_file():
            raise SystemExit("CUDA operator library missing: " + str(operators / expected_ops))
        for p in operators.rglob("*"):
            if p.is_file() and (shared(p) or "licenses" in p.parts):
                ort_payload["onnx_ops/" + p.relative_to(operators).as_posix()] = p.read_bytes()
    output = ROOT / "build/java-sdk"
    output.mkdir(parents=True, exist_ok=True)
    ort_jar = write_bundle(
        output,
        "rapid-doc-ort",
        args.ort_version,
        classifier,
        platform,
        ort_payload,
        "rapiddoc-ort",
        runtime_version=actual_version,
        device="cuda" if args.cuda else "cpu",
    )
    runtime_pom = output / f"rapid-doc-ort-{args.ort_version}.pom"
    runtime_pom.write_text(
        (ROOT / "bindings/java/ort/pom.xml")
        .read_text()
        .replace("<version>1.23.2</version>", f"<version>{args.ort_version}</version>"),
        encoding="utf-8",
    )
    mvn = shutil.which("mvn.cmd" if windows else "mvn")
    if not mvn:
        raise SystemExit("Maven not found")
    maven = [mvn] + (["-s", str(args.maven_settings.resolve())] if args.maven_settings else [])

    def install(jar, pom):
        run(
            maven
            + [
                "org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file",
                "-Dfile=" + str(jar),
                "-DpomFile=" + str(pom),
                "-Dpackaging=jar",
                "-Dclassifier=" + classifier,
            ]
        )

    install(ort_jar, runtime_pom)
    if args.ort_only:
        print("Installed independent ORT into local Maven:", ort_jar)
        return
    files = {jni: (ROOT / "target/release" / jni).read_bytes()}
    for name in (shared_library("rapiddoc_pdfium"),):
        p = runtime / name
        if p.is_file():
            files[name] = p.read_bytes()
    if not any("pdfium" in n and shared(Path(n)) for n in files):
        raise SystemExit("Prepared isolated PDFium library missing")
    for p in (runtime / "geos").iterdir():
        if p.is_file() and (shared(p) or p.name.startswith("LICENSE") or p.name.endswith(".tar.bz2")):
            files["geos/" + p.name] = p.read_bytes()
    for directory, prefix in [
        (runtime / "licenses", "licenses/pdfium"),
        (ROOT / "bindings/python/python/rapid_doc_rust/licenses", "licenses/sdk"),
    ]:
        for p in directory.rglob("*"):
            if p.is_file():
                files[prefix + "/" + p.relative_to(directory).as_posix()] = p.read_bytes()
    # JNI's runtime-only dependencies carry their own compact notice set.
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--locked", "--offline", "--format-version", "1"], cwd=ROOT
        )
    )
    tree = subprocess.check_output(
        [
            "cargo",
            "tree",
            "--locked",
            "--offline",
            "-p",
            "rapiddoc-java",
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
    selected = {
        (m[1], m[2]) for line in tree.splitlines() if (m := re.match(r"^(\S+) v(\S+)", line))
    }
    python_tree = subprocess.check_output(
        [
            "cargo",
            "tree",
            "--locked",
            "--offline",
            "-p",
            "rapiddoc-python",
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
    python_selected = {
        (m[1], m[2]) for line in python_tree.splitlines() if (m := re.match(r"^(\S+) v(\S+)", line))
    }
    from generate_release_notice_policy import selected_files
    from shared_license_terms import pack_shared

    notices = {}
    for package in metadata["packages"]:
        key = (package["name"], package["version"])
        if key in selected - python_selected and package.get("source"):
            source, keep = selected_files(package)
            if not keep:
                raise SystemExit("Missing JNI license: " + str(key))
            for name in keep:
                notices[package["name"] + "/" + name] = source[name]
    text, index = pack_shared(notices)
    files["licenses/jni/THIRD_PARTY_NOTICES.txt"] = text
    files["licenses/jni/third-party-notices.json"] = index
    import xml.etree.ElementTree as ET
    version = ET.parse(ROOT / "bindings/java/pom.xml").getroot().findtext(
        "{http://maven.apache.org/POM/4.0.0}version"
    )
    jar = write_bundle(output, "rapid-doc-native", version, classifier, platform, files, "native")
    install(jar, ROOT / "bindings/java/native/pom.xml")
    command = maven + [
        "-f",
        "bindings/java/pom.xml",
        "install",
        "-Drapid-doc.native.classifier=" + classifier,
        "-Drapid-doc.ort.version=" + args.ort_version,
    ]
    if args.skip_tests:
        command += ["-DskipTests"]
    run(command)
    print("Installed Java SDK and native runtime into local Maven:", jar)


if __name__ == "__main__":
    main()
