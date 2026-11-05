"""Lossless, readable consolidation of generated third-party notice copies.

Original source/vendor licenses and LGPL source offers are not removed. Byte
offsets and SHA-256 let provenance checks resolve the former individual paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

TEXT = "THIRD_PARTY_NOTICES.txt"
INDEX = "third-party-notices.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_encodings(entries):
    readable, separate = {}, {}
    for name, data in entries.items():
        try:
            data.decode('utf8')
        except UnicodeDecodeError:
            separate[name] = data
        else:
            readable[name] = data
    return readable, separate


def unpack(text: bytes, index: bytes) -> dict[str, bytes]:
    metadata = json.loads(index)
    if metadata.get('schema')=='rapiddoc.notice-bundle.v2':
        from shared_license_terms import unpack_shared
        return unpack_shared(text,index)
    if metadata.get("schema") != "rapiddoc.notice-bundle.v1":
        raise ValueError("Unknown third-party notice bundle schema")
    result = {}
    for entry in metadata["entries"]:
        start, end = entry["start"], entry["end"]
        if not 0 <= start <= end <= len(text):
            raise ValueError("Invalid notice byte offsets")
        data = text[start:end]
        if sha(data) != entry["sha256"]:
            raise ValueError("Third-party notice bytes changed")
        for path in entry["original_paths"]:
            parts = PurePosixPath(path)
            if parts.is_absolute() or ".." in parts.parts or path in result:
                raise ValueError(f"Invalid/duplicate notice path: {path}")
            result[path] = data
    if not result:
        raise ValueError("An empty notice bundle is not allowed")
    return result


def pack(entries: dict[str, bytes]) -> tuple[bytes, bytes]:
    groups = {}
    for path, data in sorted(entries.items()):
        data.decode("utf8")  # Do not silently transcode any original license.
        group = groups.setdefault(sha(data), {"data": data, "paths": []})
        group["paths"].append(path)
    out = bytearray(
        b"RapidDoc third-party copyright, license and attribution notices\n"
        b"Identical texts are reproduced once; every original component/path is listed.\n"
        b"Original notice contents are preserved verbatim; this does not relicense them.\n"
    )
    records = []
    for checksum, group in sorted(groups.items(), key=lambda item: item[1]["paths"][0]):
        out.extend(b"\n" + b"=" * 78 + b"\n")
        out.extend(("Components / original paths:\n" + "\n".join(group["paths"]) + "\n").encode())
        out.extend(("SHA-256: " + checksum + "\n\n").encode())
        start = len(out)
        out.extend(group["data"])
        records.append(dict(sha256=checksum, original_paths=group["paths"], start=start, end=len(out)))
        out.extend(b"\n")
    index = json.dumps(dict(schema="rapiddoc.notice-bundle.v1", entries=records),
                       ensure_ascii=False, indent=2).encode() + b"\n"
    if unpack(bytes(out), index) != entries:
        raise ValueError("Notice consolidation did not preserve every original byte")
    return bytes(out), index


def read_tree(root: Path) -> dict[str, bytes]:
    entries = {}
    consumed = set()
    for index in sorted(root.rglob(INDEX), key=lambda p: (len(p.parts), p.as_posix())):
        text = index.with_name(TEXT)
        if not text.is_file():
            raise ValueError(f"Notice index has no text: {index}")
        prefix = index.parent.relative_to(root)
        for name, data in unpack(text.read_bytes(), index.read_bytes()).items():
            entries[(prefix / name).as_posix()] = data
        consumed.update((index, text))
    for path in sorted(root.rglob("*")):
        if path.is_file() and path not in consumed:
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            entries[name] = data
    return entries


def compact(root: Path, *, selected_prefixes: tuple[str, ...] = (), entries=None, shared_terms=False) -> dict:
    root = root.resolve()
    workspace = Path(__file__).resolve().parents[2]
    if not root.is_relative_to(workspace):
        raise ValueError("Refuse to clean a notice tree outside this workspace")
    entries = read_tree(root) if entries is None else dict(entries)
    if selected_prefixes:
        entries = {name: data for name, data in entries.items()
                   if name.startswith(selected_prefixes)}
    if not entries:
        raise ValueError(f"No notices selected: {root}")
    entries, separate = split_encodings(entries)
    if shared_terms:
        from shared_license_terms import pack_shared
        text,index=pack_shared(entries)
    else:
        text, index = pack(entries)
    old_files = [p for p in root.rglob("*") if p.is_file()]
    removable = [p for p in old_files if p.relative_to(root).as_posix() not in separate
                 and (p.name in (TEXT, INDEX) or not selected_prefixes
                      or p.relative_to(root).as_posix().startswith(selected_prefixes))]
    # Resolve and check every target before writing or deleting. No recursive
    # delete, symlinks, computed parent directory, or external directory follows.
    for path in removable:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe notice cleanup target: {path}")
    (root / TEXT).write_bytes(text)
    (root / INDEX).write_bytes(index)
    assert unpack((root / TEXT).read_bytes(), (root / INDEX).read_bytes()) == entries
    for path in removable:
        if path not in (root / TEXT, root / INDEX):
            path.unlink()
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    return dict(before=len(old_files), after=sum(p.is_file() for p in root.rglob("*")),
                original_paths=len(entries), unique_texts=len(json.loads(index)["entries"]))


def read_notice(path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    for parent in path.parents:
        index = parent / INDEX
        if index.is_file():
            entries = unpack((parent / TEXT).read_bytes(), index.read_bytes())
            name = path.relative_to(parent).as_posix()
            if name in entries:
                return entries[name]
    raise FileNotFoundError(path)


def compact_assets(assets: dict[str, bytes], root: str, transform=None, shared_terms=False) -> None:
    """Consolidate a wheel's notice subtree, including an already bundled tree."""
    names = [name for name in assets if name.startswith(root)]
    if not names:
        return
    entries = {}
    consumed = set()
    for name in sorted(names, key=lambda n: (n.count('/'), n)):
        if name.endswith("/" + INDEX):
            prefix = name[:-len(INDEX)]
            for original, data in unpack(assets[prefix + TEXT], assets[name]).items():
                entries[prefix[len(root):] + original] = data
            consumed.update((name, prefix + TEXT))
    for name in names:
        if name not in consumed:
            entries[name[len(root):]] = assets[name]
    if transform is not None:
        entries = transform(entries)
    entries, separate = split_encodings(entries)
    if shared_terms:
        from shared_license_terms import pack_shared
        text,index=pack_shared(entries)
    else:
        text, index = pack(entries)
    for name in names:
        del assets[name]
    assets[root + TEXT], assets[root + INDEX] = text, index
    assets.update({root+name: data for name, data in separate.items()})
