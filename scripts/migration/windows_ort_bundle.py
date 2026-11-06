"""Build a verified Windows ORT private CRT bundle, never at application startup.

Only the derived ORT copy is resource-edited; its Microsoft signature no longer
applies. CRT binaries retain their signatures. Publishers may sign the derived
DLL before packaging. Do not use System32 as a redistributable source.
"""

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from native_platform import host_architecture
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

LIBRARIES = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
)


def text_hash(path):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    size = struct.unpack_from("<H", data, pe + 20)[0]
    count = struct.unpack_from("<H", data, pe + 6)[0]
    for index in range(count):
        section = pe + 24 + size + index * 40
        if data[section : section + 8].rstrip(b"\0") == b".text":
            length, offset = struct.unpack_from("<II", data, section + 16)
            return hashlib.sha256(data[offset : offset + length]).hexdigest()
    raise ValueError("ORT .text section not found")


def remove_private_manifest(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    kernel.BeginUpdateResourceW.restype = wintypes.HANDLE
    kernel.UpdateResourceW.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.WORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.UpdateResourceW.restype = wintypes.BOOL
    kernel.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
    kernel.EndUpdateResourceW.restype = wintypes.BOOL
    handle = kernel.BeginUpdateResourceW(str(path), False)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    # Verified ORT 1.23.0/1.23.2 resource ID/language, not a generic PE patcher.
    if not kernel.UpdateResourceW(handle, ctypes.c_void_p(24), ctypes.c_void_p(2), 1033, None, 0):
        error = ctypes.get_last_error()
        kernel.EndUpdateResourceW(handle, True)
        raise ctypes.WinError(error)
    if not kernel.EndUpdateResourceW(handle, False):
        raise ctypes.WinError(ctypes.get_last_error())


PRIVATE = "RapidDoc.Runtime.PrivateCRT"


def require_architecture(path, architecture):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    machine = struct.unpack_from("<H", data, pe + 4)[0]
    # Some official Arm CRT hybrids retain an AMD64 COFF machine value.
    expected = {"x86_64": (0x8664,), "aarch64": (0xAA64, 0xA64E)}[architecture]
    if architecture == "aarch64" and machine == 0x8664 and data[pe : pe + 4] == b"PE\0\0":
        # Use the installed MSVC tool to distinguish ARM64X from ordinary x64;
        # a machine-value whitelist alone rejects vcruntime140_1.dll from VS.
        headers = subprocess.check_output(["dumpbin", "/headers", str(path)], text=True)
        if "(ARM64X)" in headers:
            return
    if data[pe : pe + 4] != b"PE\0\0" or machine not in expected:
        raise RuntimeError(f"Expected Windows {architecture} PE: {path}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_version(path):
    version = ctypes.WinDLL("version", use_last_error=True)
    version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
    version.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    version.GetFileVersionInfoW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    version.VerQueryValueW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.UINT),
    ]
    size = version.GetFileVersionInfoSizeW(str(path), None)
    buffer = ctypes.create_string_buffer(size)
    if not size or not version.GetFileVersionInfoW(str(path), 0, size, buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    pointer, length = ctypes.c_void_p(), wintypes.UINT()
    if not version.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    words = ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD))
    return (words[2] >> 16, words[2] & 65535, words[3] >> 16, words[3] & 65535)


def find_crt(architecture):
    redist_arch = "arm64" if architecture == "aarch64" else "x64"
    explicit = os.environ.get("RAPIDDOC_CRT_REDIST_DIR")
    if explicit:
        return Path(explicit).resolve(strict=True)
    where = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installations = subprocess.check_output(
        [str(where), "-all", "-products", "*", "-property", "installationPath"], text=True
    ).splitlines()
    candidates = [
        path
        for installation in installations
        for path in (Path(installation) / "VC/Redist/MSVC").glob(f"*/{redist_arch}/Microsoft.VC*.CRT")
        if (path / "msvcp140.dll").is_file()
    ]
    if not candidates:
        raise RuntimeError(f"Set RAPIDDOC_CRT_REDIST_DIR to a licensed VS {redist_arch} CRT redist directory")
    return max(candidates, key=lambda p: file_version(p / "msvcp140.dll"))


def bundle_files(ort_directory, crt_directory=None, signer=None):
    """Return complete relative-path/bytes payload for a Windows platform package.

    signer, when supplied, signs only the temporary derived DLL before hashing.
    Inputs are never modified. No persistent DLL cache or runtime network access.
    """
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("Private CRT packaging requires 64-bit Windows")
    source = ort_directory.resolve(strict=True) / "onnxruntime.dll"
    architecture = host_architecture()
    require_architecture(source, architecture)
    library = ctypes.WinDLL(str(source))
    library.OrtGetApiBase.restype = ctypes.POINTER(ctypes.c_void_p)
    version_function = ctypes.WINFUNCTYPE(ctypes.c_char_p)(library.OrtGetApiBase()[1])
    version = version_function().decode("ascii")
    if version not in {"1.23.0", "1.23.2"}:
        raise RuntimeError(f"Private CRT supports validated ORT 1.23.0/1.23.2 only, got {version}")
    crt = (crt_directory or find_crt(architecture)).resolve(strict=True)
    if "system32" in (part.lower() for part in crt.parts):
        raise RuntimeError("Use the Visual Studio redistributable source, not System32")
    crt_version = file_version(crt / "msvcp140.dll")
    # The Windows Arm runner ships VS 2022 (14.44); x64 retains its tested 14.51
    # baseline. Native installation/inference checks run before keeping artifacts.
    supported_crt = {(14, 44), (14, 51)} if architecture == "aarch64" else {(14, 51)}
    if crt_version[:2] not in supported_crt:
        raise RuntimeError(f"Unsupported private CRT version for {architecture}: {crt_version}")
    files = {}
    # Arm redists have architecture-specific helpers (for example the threads
    # runtime). Retain the complete redistributable DLL set rather than the x64 list.
    libraries = LIBRARIES if architecture == "x86_64" else tuple(
        sorted(path.name for path in crt.glob("*.dll"))
    )
    for required in ("msvcp140.dll", "vcruntime140.dll"):
        if required not in libraries:
            raise RuntimeError("Required CRT library missing: " + required)
    for name in libraries:
        path = crt / name
        require_architecture(path, architecture)
        if file_version(path) != crt_version:
            raise RuntimeError("Mixed CRT versions: " + str(path))
        files[name] = path.read_bytes()
    source_hash = sha(source)
    with tempfile.TemporaryDirectory(prefix="rapiddoc-ort-") as temporary:
        target = Path(temporary) / "onnxruntime.dll"
        shutil.copy2(source, target)
        remove_private_manifest(target)
        if text_hash(source) != text_hash(target):
            raise RuntimeError("ORT code section changed")
        if signer:
            signer(target)
            if text_hash(source) != text_hash(target):
                raise RuntimeError("Signing changed the ORT code section")
        files[target.name] = target.read_bytes()
    if sha(source) != source_hash:
        raise RuntimeError("Source ORT changed during packaging")
    # Providers stay beside the main ORT DLL, including CUDA's shared provider.
    for path in ort_directory.glob("onnxruntime_providers*.dll"):
        files[path.name] = path.read_bytes()
    manifest_arch = "arm64" if architecture == "aarch64" else "amd64"
    identity = (
        'type="win32" name="RapidDoc.Runtime.PrivateCRT" '
        f'version="1.0.0.0" processorArchitecture="{manifest_arch}"'
    )
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n'
    )
    entries = "".join(f'<file name="{name}" />\n' for name in files)
    files[PRIVATE + ".manifest"] = (
        header + f"<assemblyIdentity {identity} />\n" + entries + "</assembly>\n"
    ).encode()
    files["rapiddoc-ort.manifest"] = (
        header
        + '<assemblyIdentity type="win32" name="RapidDoc.OrtLoader" version="1.0.0.0" />\n'
        + f"<dependency><dependentAssembly><assemblyIdentity {identity} /></dependentAssembly></dependency>\n"
        + "</assembly>\n"
    ).encode()
    receipt = dict(
        schema=1,
        architecture=architecture,
        ort_version=version,
        crt_version=list(crt_version),
        source_ort_sha256=source_hash,
        source_crt_sha256={name: sha(crt / name) for name in libraries},
        derived_ort_sha256=hashlib.sha256(files["onnxruntime.dll"]).hexdigest(),
        ort_text_sha256=text_hash(source),
        modified_resource="RT_MANIFEST/2/1033 removed from ORT copy",
        microsoft_signature_preserved=False,
        publisher_signing_requested=signer is not None,
    )
    files["provenance.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    files["NOTICE.txt"] = (
        f"Microsoft Visual C++ {'.'.join(map(str, crt_version))} {architecture} runtime files are redistributed unchanged from "
        "the Visual Studio REDIST directory under the publisher's Visual Studio license.\n"
        "https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files\n"
        f"ORT is a derived {version} binary: only its embedded activation manifest was removed; "
        "the executable .text section is unchanged. Microsoft's ORT signature does not "
        "apply to the derived DLL. See licenses/ort for ORT terms and notices.\n"
    ).encode()
    return {PRIVATE + "/" + name: data for name, data in files.items()}
