"""Offline release guards; never authenticate or upload to a registry."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/release"))
import common
import crates
import npm_packages
import publish_central
import publish_npm

VERSION = common.version()


@pytest.mark.parametrize(
    "os_name,machine,expected,library",
    [("win32", "AMD64", "windows-x86_64", "test.dll"),
     ("linux", "x86_64", "linux-x86_64", "libtest.so"),
     ("darwin", "x86_64", "macos-x86_64", "libtest.dylib"),
     ("darwin", "arm64", "macos-aarch64", "libtest.dylib")],
)
def test_native_platform_names_and_library_extensions(monkeypatch, os_name, machine, expected, library):
    import native_platform
    monkeypatch.setattr(native_platform.sys, "platform", os_name)
    monkeypatch.setattr(native_platform.platform, "machine", lambda: machine)
    assert native_platform.host_platform() == expected
    assert native_platform.shared_library("test") == library
    assert native_platform.is_shared_library(Path(library))
    if os_name == "darwin":
        with pytest.raises(ValueError, match="CPU"):
            native_platform.host_platform(cuda=True)


def test_versions_and_publish_tag_are_checked_before_upload(monkeypatch):
    common.check_version(VERSION)
    for value in ["../0.1.0", "0.1.0; echo invalid", "00.1.0", "0.1.0-rc1", "1.0.0"]:
        with pytest.raises(ValueError):
            common.check_version(value)
    for ref in ["refs/heads/main", "refs/tags/v0.1.0", "refs/tags/sdk-v9999.0.0"]:
        monkeypatch.setenv("GITHUB_REF", ref)
        with pytest.raises(ValueError, match="matching sdk-v"):
            common.check_version(VERSION, publishing=True)
    monkeypatch.setenv("GITHUB_REF", f"refs/tags/sdk-v{VERSION}")
    common.check_version(VERSION, publishing=True)


def test_payload_integrity_rejects_corruption_and_directory_escape(tmp_path):
    (tmp_path / "data").write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    common.checked_files(tmp_path, {"data": digest})
    for files in [{"missing": digest}, {"../data": digest}, {"data": "0" * 64}]:
        with pytest.raises(ValueError):
            common.checked_files(tmp_path, files)


@pytest.mark.parametrize(
    "platform,package,os_name",
    [
        ("windows-x86_64", "rapid-doc-win32-x64", "win32"),
        ("linux-x86_64", "rapid-doc-linux-x64-gnu", "linux"),
        ("linux-x86_64-cuda", "rapid-doc-linux-x64-gnu-cuda", "linux"),
        ("macos-x86_64", "rapid-doc-darwin-x64", "darwin"),
        ("macos-aarch64", "rapid-doc-darwin-arm64", "darwin"),
    ],
)
def test_npm_native_packages_have_distinct_names_and_retain_verified_payload(
    tmp_path, platform, package, os_name
):
    source = tmp_path / "source"
    (source / "runtime").mkdir(parents=True)
    (source / "runtime/rapid_doc_node.node").write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    (source / "runtime/manifest.json").write_text(
        json.dumps({"platform": platform, "files": {"rapid_doc_node.node": digest}})
    )
    out = tmp_path / "out"
    npm_packages.assemble(source, out)
    p = json.loads((out / "package.json").read_text())
    assert p["name"] == package and p["os"] == [os_name]
    assert p["cpu"] == (["arm64"] if platform == "macos-aarch64" else ["x64"])
    assert (out / "runtime/rapid_doc_node.node").read_bytes() == b"fixture"


def test_npm_root_only_auto_installs_cpu_platforms(tmp_path):
    npm_packages.root_package(tmp_path / "root")
    p = json.loads((tmp_path / "root/package.json").read_text())
    assert p["name"] == "rapid-doc"
    assert p["optionalDependencies"] == {
        "rapid-doc-win32-x64": VERSION,
        "rapid-doc-linux-x64-gnu": VERSION,
        "rapid-doc-darwin-x64": VERSION,
        "rapid-doc-darwin-arm64": VERSION,
    }
    assert (tmp_path / "root/platform.js").exists()
    assert not (tmp_path / "root/runtime").exists()


def test_npm_upload_plan_requires_both_platforms_and_publishes_loader_last(
    tmp_path, monkeypatch
):
    import io
    import tarfile

    monkeypatch.setattr(publish_npm, "ROOT", tmp_path)

    def tarball(directory, name):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".tgz")
        data = json.dumps({"name": name, "version": VERSION}).encode()
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo("package/package.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        return path

    tarball(tmp_path / "build/release/npm-root", "rapid-doc")
    with pytest.raises(ValueError, match="four CPU"):
        publish_npm.packages()
    for platform, package in [
        ("windows", "rapid-doc-win32-x64"),
        ("linux", "rapid-doc-linux-x64-gnu"),
        ("macos-x86_64", "rapid-doc-darwin-x64"),
        ("macos-aarch64", "rapid-doc-darwin-arm64"),
    ]:
        tarball(tmp_path / f"artifacts/sdk-{platform}/npm", package)
    plan = publish_npm.packages()
    assert len(plan) == 5 and plan[-1][0] == "rapid-doc"
    tarball(tmp_path / "artifacts/sdk-duplicate/npm", "rapid-doc-win32-x64")
    with pytest.raises(ValueError, match="duplicate"):
        publish_npm.packages()


def test_crates_preflight_accepts_versioned_forks_without_running_cargo(monkeypatch):
    monkeypatch.setattr(
        crates, "run", lambda *a, **k: pytest.fail("readiness must not publish")
    )
    assert crates.readiness() == []


def test_crates_preflight_rejects_wrong_registry_versions_and_local_paths(
    tmp_path, monkeypatch
):
    import shutil

    for name in crates.ORDER:
        source = crates.package_root(name)
        target = tmp_path / source.relative_to(ROOT)
        target.mkdir(parents=True)
        shutil.copyfile(source / "Cargo.toml", target / "Cargo.toml")
    monkeypatch.setattr(crates, "ROOT", tmp_path)
    manifest = crates.package_root("rapiddoc-pdf") / "Cargo.toml"
    source = manifest.read_text()
    manifest.write_text(source.replace('version = "=0.42.0"', 'version = "=0.41.0"'))
    assert any("registry version differs" in issue for issue in crates.readiness())
    manifest.write_text(source.replace("../vendor/lopdf-f64", "../vendor/wrong-fork"))
    assert any("path differs" in issue for issue in crates.readiness())
    manifest.write_text(source.replace('version = "=0.42.0", ', ""))
    assert any("needs a registry version" in issue for issue in crates.readiness())


def test_central_rejects_incomplete_or_conflicting_artifacts(tmp_path):
    artifacts = tmp_path / "artifacts"
    (artifacts / "sdk-linux/java").mkdir(parents=True)
    (artifacts / "sdk-windows/java").mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete"):
        publish_central.assemble(artifacts, tmp_path / "central")
    for name, content in [("sdk-linux", b"a"), ("sdk-windows", b"b")]:
        (
            artifacts / name / f"java/rapid-doc-native-{VERSION}-linux-x86_64.jar"
        ).write_bytes(content)
    with pytest.raises(ValueError, match="Conflicting"):
        publish_central.assemble(artifacts, tmp_path / "central")


def test_central_merges_common_poms_with_platform_line_endings(tmp_path):
    artifacts = tmp_path / "artifacts"
    for platform, text in [("windows", b"<project>\r\n</project>\r\n"),
                           ("macos", b"<project>\n</project>\n")]:
        directory = artifacts / platform / "java"
        directory.mkdir(parents=True)
        (directory / f"rapid-doc-{VERSION}.pom").write_bytes(text)
    with pytest.raises(ValueError, match="Incomplete"):
        publish_central.assemble(artifacts, tmp_path / "central")
