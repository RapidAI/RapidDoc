"""Regression checks for ORT archives whose aliases must not become extra binaries."""

import io
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/release"))
import runtime
from native_platform import ort_library_files


@pytest.mark.parametrize("main,aliases,provider", [
    ("libonnxruntime.so.1.23.2", ["libonnxruntime.so", "libonnxruntime.so.1"],
     "libonnxruntime_providers_shared.so"),
    ("libonnxruntime.1.23.2.dylib", ["libonnxruntime.dylib"],
     "libonnxruntime_providers_shared.dylib"),
])
def test_extract_preserves_real_libraries_and_notices_without_expanding_links(
    tmp_path, main, aliases, provider
):
    source = tmp_path / "ort.tgz"
    with tarfile.open(source, "w:gz") as archive:
        # Upstream aliases can precede the file they point at.
        for alias in aliases:
            entry = tarfile.TarInfo("ort/lib/" + alias)
            entry.type = tarfile.SYMTYPE
            entry.linkname = main
            archive.addfile(entry)
        for name, data in [("lib/" + main, b"main"), ("lib/" + provider, b"provider"),
                           ("LICENSE", b"license"), ("ThirdPartyNotices.txt", b"notices"),
                           ("lib/libonnxruntime.a", b"static"), ("include/header.h", b"header")]:
            entry = tarfile.TarInfo("ort/" + name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    destination = tmp_path / "runtime"
    destination.mkdir()
    runtime.extract_ort(source, destination)
    assert {p.name for p in destination.iterdir()} == {
        main, provider, "LICENSE", "ThirdPartyNotices.txt"
    }
    assert (destination / main).read_bytes() == b"main"
    assert (destination / provider).read_bytes() == b"provider"


def test_legacy_bundle_drops_identical_aliases_but_retains_distinct_libraries(tmp_path):
    for name, data in [("libonnxruntime.so.1.23.2", b"main"),
                       ("libonnxruntime.so.1", b"main"),
                       ("libonnxruntime.so", b"custom"),
                       ("libonnxruntime_providers_shared.so", b"provider")]:
        (tmp_path / name).write_bytes(data)
    selected = {p.name for p in ort_library_files(tmp_path)}
    assert selected == {"libonnxruntime.so.1.23.2", "libonnxruntime.so",
                        "libonnxruntime_providers_shared.so"}


def test_external_soname_only_bundle_remains_usable(tmp_path):
    (tmp_path / "libonnxruntime.so.1").write_bytes(b"main")
    assert [p.name for p in ort_library_files(tmp_path)] == ["libonnxruntime.so.1"]


def test_windows_provider_payload_remains_intact(tmp_path):
    names = ["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"]
    for name in names:
        (tmp_path / name).write_bytes(b"dll")
    assert {p.name for p in ort_library_files(tmp_path)} == set(names)


def test_windows_archive_still_delivers_providers_and_notices(tmp_path):
    source = tmp_path / "ort.zip"
    files = {"lib/onnxruntime.dll": b"main", "lib/onnxruntime_providers_cuda.dll": b"cuda",
             "LICENSE": b"license", "ThirdPartyNotices.txt": b"notices",
             "lib/onnxruntime.pdb": b"symbols"}
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in files.items():
            archive.writestr("ort/" + name, data)
    destination = tmp_path / "runtime"
    destination.mkdir()
    runtime.extract_ort(source, destination)
    assert {p.name for p in destination.iterdir()} == {
        "onnxruntime.dll", "onnxruntime_providers_cuda.dll", "LICENSE", "ThirdPartyNotices.txt"
    }
