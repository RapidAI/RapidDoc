"""Verify a mechanical Rust reorganization against the accepted source snapshot."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "build/pdf-structure-before-20261030"


def tokens(text):
    """Ignore trivia while preserving literal contents, identifiers and punctuation."""
    result = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
        elif text.startswith("/*", i):
            i += 2
            depth = 1
            while depth:
                assert i < len(text), "Unclosed comment"
                if text.startswith("/*", i): depth += 1; i += 2
                elif text.startswith("*/", i): depth -= 1; i += 2
                else: i += 1
        else:
            raw = re.match(r'(?:b|c)?r(\#*)"', text[i:])
            literal = re.match(r'"(?:\\.|[^"\\])*"|\'(?:\\(?:u\{[^}]*\}|x[0-9a-fA-F]{2}|.)|[^\'\\\n])\'', text[i:], re.S)
            word = re.match(r"\w+", text[i:])
            if raw:
                end = text.find('"' + raw[1], i + raw.end())
                assert end >= 0, "Unclosed raw string"
                end += 1 + len(raw[1])
            elif literal: end = i + literal.end()
            elif word: end = i + word.end()
            else: end = i + 1
            result.append(text[i:end])
            i = end
    # rustfmt inserts/removes optional trailing commas when lines wrap.
    return [t for i, t in enumerate(result) if not (t == "," and i + 1 < len(result) and result[i + 1] in (")", "]", "}"))]


def normalized(path):
    text = path.read_text("utf-8")
    text = re.sub(r'#\[path\s*=\s*"[^"\n]+"\]\s*', "", text)
    text = re.sub(r"(?m)^(?:pub )?mod \w+;\s*", "", text)
    text = text.replace("pub(in crate::processing)", "pub(in crate::legacy_model)")
    text = text.replace("pub(in crate::page_assembly)", "pub(in crate::legacy_page)")
    text = re.sub(r"pub use (?:processing as legacy_model|page_assembly as legacy_page);", "", text)
    def resource(match):
        resource = path.parent / match[2]
        if not resource.exists() and path.is_relative_to(BEFORE / "formatted"):
            relative = path.parent.relative_to(BEFORE / "formatted") / match[2]
            resource = BEFORE / "source" / relative
            if not resource.exists():
                resource = ROOT / relative
        elif not resource.exists() and path.is_relative_to(BEFORE / "source"):
            resource = ROOT / path.parent.relative_to(BEFORE / "source") / match[2]
        data = resource.read_bytes()
        return f'{match[1]}!("sha256:{hashlib.sha256(data).hexdigest()}")'
    return tokens(re.sub(r'(include_(?:str|bytes))!\("([^"]+)"\)', resource, text))


def binding_items(path):
    text = path.read_text("utf-8").replace("pub(super) ", "")
    lines = text.splitlines(keepends=True)
    declarations = [(i, m[1], m[2]) for i, line in enumerate(lines) if (m := re.match(r"^(fn|struct|impl)\s+(\w+)", line))]
    starts = []
    for index, (line, _, _) in enumerate(declarations):
        if index == 0:
            # Imports and file docs are outside the first item's attributes.
            start = max([0] + [i + 1 for i in range(line) if lines[i].startswith("use ") or lines[i].startswith("type ") or lines[i].startswith("mod ")])
        else:
            previous = declarations[index - 1][0]
            start = max([previous] + [i for i in range(previous + 1, line) if lines[i].startswith("}")]) + 1
        starts.append(start)
    return {(kind, name): tokens("".join(lines[starts[i]:starts[i+1] if i+1<len(starts) else len(lines)]))
            for i, (_, kind, name) in enumerate(declarations)}


def main():
    frozen = json.loads((BEFORE / "manifest.json").read_text("utf-8"))
    moves = json.loads((ROOT / "build/pdf-structure-moves-20261030.json").read_text("utf-8"))
    report = dict(identical_implementations=[], identical_resources=[], mismatches=[])
    for source in frozen["files"]:
        if source == "bindings/python/src/lib.rs":
            continue
        original = BEFORE / ("formatted" if source.endswith(".rs") else "source") / source
        actual = ROOT / moves["moves"].get(source, source)
        if not actual.exists():
            report["mismatches"].append(dict(source=source, reason="missing")); continue
        key = "identical_implementations" if source.endswith(".rs") else "identical_resources"
        same = normalized(original) == normalized(actual) if source.endswith(".rs") else original.read_bytes() == actual.read_bytes()
        if same: report[key].append(dict(source=source, target=actual.relative_to(ROOT).as_posix()))
        else: report["mismatches"].append(dict(source=source, target=actual.relative_to(ROOT).as_posix()))
    original = binding_items(BEFORE / "formatted/bindings/python/src/lib.rs")
    current = {}
    for file in ["lib", "pdf", "assets", "tables", "ocr", "formula", "models", "diagnostics"]:
        current.update(binding_items(ROOT / f"bindings/python/src/{file}.rs"))
    binding_differences = [list(key) for key in original if original[key] != current.get(key)]
    report.update(binding_items=len(original), binding_differences=binding_differences)
    (ROOT / "build/pdf-structure-source-audit-20261030.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print("Identical implementations", len(report["identical_implementations"]), "resources", len(report["identical_resources"]), "binding items", len(original))
    print("Mismatches", report["mismatches"], binding_differences)
    return 1 if report["mismatches"] or binding_differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
