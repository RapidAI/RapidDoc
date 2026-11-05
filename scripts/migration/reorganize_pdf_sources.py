"""One-time, manifest-recorded source moves for the accepted PDF implementation."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOVES = {}


def checked(path):
    path = path.resolve()
    assert path.is_relative_to(ROOT), path
    return path


def move(source, target):
    source, target = checked(source), checked(target)
    assert source.exists() and not target.exists(), (source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for child in source.rglob("*"):
            if child.is_file():
                MOVES[child.relative_to(ROOT).as_posix()] = (target / child.relative_to(source)).relative_to(ROOT).as_posix()
    else:
        MOVES[source.relative_to(ROOT).as_posix()] = target.relative_to(ROOT).as_posix()
    source.rename(target)


def rule_domain(name):
    if name.startswith(("table_", "tiled_", "filled_grid", "closed_grid", "closed_rule", "prose_rule", "rule_")):
        return "table"
    if name in {"formulas", "native_vector_formulas", "native_formula_text", "native_formula_components"}:
        return "formula"
    if name in {"native_ocr", "ocr_results", "crops"}:
        return "ocr"
    if name.startswith(("bar_", "graphic_", "concentric_", "drawing_", "isolated_", "path_", "raster_", "stacked_", "diagram_", "strong_")) or name == "image_regions":
        return "graphics"
    if name in {"text_noise", "source_lines", "native_lines", "native_titles", "native_marginals", "native_references", "native_text_roles", "native_footnotes", "native_glyph_weight", "native_text_assembly", "native_document_text"}:
        return "text"
    if name in {"native_document", "native_pdf_document", "native_finalization", "native_final_output", "native_inline_document", "inline_output", "inline_evidence", "code_containers", "page_containers", "native_page_preparation"}:
        return "document"
    assert name in {"native_geometry", "native_layout", "native_precontainers", "native_adapter", "native_model_normalization", "native_visual_annotations", "native_grouped_order"}, name
    return "layout"


def processing():
    base = ROOT / "rust/rapiddoc/src"
    original = base / "legacy_model"
    target = base / "processing"
    # Record original paths directly, even though the directory moves first.
    move(original, target)
    move(base / "legacy_model.rs", target / "mod.rs")
    source = (target / "mod.rs").read_text("utf-8")
    entries = re.findall(r"(?m)^((?:pub )?mod (\w+);)$", source)
    for declaration, name in entries:
        group = rule_domain(name)
        move(target / (name + ".rs"), target / group / (name + ".rs"))
        if (target / name).is_dir():
            move(target / name, target / group / name)
        source = source.replace(declaration, f'#[path = "{group}/{name}.rs"]\n{declaration}')
    # Path-selected modules use the selected file's directory for child lookup.
    # Give components with children a standard mod.rs entry, retaining includes.
    for path in sorted(target.rglob("*.rs"), key=lambda p: len(p.parts), reverse=True):
        child = path.with_suffix("")
        if not child.is_dir():
            continue
        text = re.sub(r'(include_(?:str|bytes)!\(")' + re.escape(path.stem) + '/', r'\1', path.read_text("utf-8"))
        destination = child / "mod.rs"
        move(path, destination)
        destination.write_text(text, "utf-8")
        source = source.replace(path.relative_to(target).as_posix(), destination.relative_to(target).as_posix())
    for name, group in [("native_line_defaults.json", "text"), ("native_model_normalization.json", "layout"), ("table_html_rules.json", "table")]:
        move(target / name, target / group / name)
    (target / "mod.rs").write_text(source, "utf-8")
    move(base / "legacy_page", base / "page_assembly")
    move(base / "legacy_page.rs", base / "page_assembly/mod.rs")
    library = base / "lib.rs"
    source = library.read_text("utf-8").replace("pub mod legacy_model;", "pub mod processing;\n/// Compatibility path for the original migration API.\npub use processing as legacy_model;")
    source = source.replace("pub mod legacy_page;", "pub mod page_assembly;\n/// Compatibility path for the original migration API.\npub use page_assembly as legacy_page;")
    library.write_text(source, "utf-8")
    for path in base.rglob("*.rs"):
        source = path.read_text("utf-8")
        canonical = source.replace("pub(in crate::legacy_model)", "pub(in crate::processing)").replace("pub(in crate::legacy_page)", "pub(in crate::page_assembly)")
        if canonical != source:
            path.write_text(canonical, "utf-8")


def inference():
    base = ROOT / "rust/rapiddoc-inference/src"
    library = base / "lib.rs"
    source = library.read_text("utf-8")
    for declaration, name in re.findall(r"(?m)^((?:pub )?mod (\w+);)$", source):
        if name.startswith("layout"):
            group = "layout"
        elif name.startswith("formula"):
            group = "formula"
        elif name.startswith(("table_", "unet", "wired_")) or name in {"blank_cell", "page_tables"}:
            group = "table"
        elif name in {"named", "registry", "shared_arena"}:
            group = "runtime"
        elif name in {"crop", "crop_mask", "pillow_resize", "orientation"}:
            group = "image"
        elif name == "seal":
            group = "seal"
        else:
            assert name in {"ctc", "detection", "detection_geometry", "detection_mask", "detection_offset", "detection_preprocess", "native_text", "numpy_argsort", "preprocess", "recognition_batch", "unicode_fix"}, name
            group = "ocr"
        entry = f"{group}/mod.rs" if name == group else f"{group}/{name.removeprefix(group + '_')}.rs"
        move(base / (name + ".rs"), base / entry)
        # Existing child directories of formula/seal stay beside their new mod.rs.
        if name != group and (base / name).is_dir():
            move(base / name, base / group / name)
        source = source.replace(declaration, f'#[path = "{entry}"]\n{declaration}')
    move(base / "unicode_fix_rules.json", base / "ocr/unicode_fix_rules.json")
    library.write_text(source, "utf-8")


def binding_domain(name):
    if name in {"python_error", "serialize", "_native"}:
        return "root"
    if name in {"FormulaDecoder", "repair_formula_latex", "fix_formula_text", "prepare_formula_input_bytes", "validate_formula_png"}:
        return "formula"
    if name in {"OnnxSession", "prepare_layout_rgb_json", "layout_rectangles_json", "prepare_orientation_rgb_json"}:
        return "models"
    if name in {"OcrLine", "run_ocr", "ocr_png_json", "parse_pdf_ocr_json", "fix_unicode_text", "crop_bgr_json", "seal_geometry_json"} or name.startswith(("prepare_model_", "prepare_recognition_", "prepare_detection_", "detection_boxes")):
        return "ocr"
    if "assets" in name or name.startswith(("materialize_", "native_pdf_")) or name in {"NativeTablePage", "NativeTextPage", "PageImage", "fill_native_text_bytes", "native_glyph_weight_json"}:
        return "assets"
    if name.startswith(("classification_", "classify_pdf_", "pdfium_", "read_pdf_", "render_pdf_", "native_classification_")) or name in {"extract_original_pdf_images_bytes", "read_native_pdf_page_json", "parse_pdf_text_json", "recover_pdf_table_json"}:
        return "pdf"
    if "table" in name.lower() or "wired" in name.lower() or name in {"blank_cell_crop_json", "match_cells_json", "prepare_unet_bgr_json", "unet_line_geometry_json", "prepare_unet_line_masks_json"}:
        return "tables"
    assert name in {"call_host_page", "legacy_host_pipeline_json", "legacy_pipeline_json", "sort_model_blocks_json", "split_paragraphs_json", "merge_text_lines", "organize_page_text_json", "validate_rect", "resolve_page_range", "deduplicate_native_characters_json"}, name
    return "diagnostics"


def bindings():
    base = ROOT / "bindings/python/src"
    library = base / "lib.rs"
    lines = library.read_text("utf-8").splitlines(keepends=True)
    declarations = [(i, m.group(1), m.group(2)) for i, line in enumerate(lines) if (m := re.match(r"^(fn|struct|impl)\s+(\w+)", line))]
    starts = []
    for index, (line, kind, name) in enumerate(declarations):
        if index == 0:
            start = next(i for i in range(line) if lines[i].startswith("#[pyfunction]"))
        else:
            previous = declarations[index - 1][0]
            end = max([previous] + [i for i in range(previous + 1, line) if lines[i].startswith("}")])
            start = end + 1
        starts.append(start)
    output = {name: [] for name in ["root", "pdf", "assets", "tables", "ocr", "formula", "models", "diagnostics"]}
    output["root"].append("".join(lines[:starts[0]]))
    mapping = {}
    for i, (_, kind, name) in enumerate(declarations):
        chunk = "".join(lines[starts[i]:starts[i + 1] if i + 1 < len(starts) else len(lines)])
        domain = binding_domain(name)
        if domain != "root" and kind != "impl":
            chunk = re.sub(r"(?m)^(fn|struct) ", r"pub(super) \1 ", chunk, count=1)
        if name == "OnnxSession" and kind == "struct":
            chunk = chunk.replace("    model:", "    pub(super) model:")
        output[domain].append(chunk)
        mapping[name] = domain
    imports = "\n".join(f"mod {name};\nuse {name}::*;" for name in output if name != "root")
    for name, chunks in output.items():
        if name == "root":
            source = "".join(chunks).replace("mod pipeline;", "mod pipeline;\n" + imports)
            source = source.replace("//! Python 3.10+ 的独立预览扩展；解析器尚未接入时不报告格式支持。", "//! Python 3.10+ extension registration; implementation lives in domain adapters.")
            library.write_text(source, "utf-8")
        else:
            (base / (name + ".rs")).write_text(f"//! {name} adapters; algorithms remain in the shared Rust crates.\nuse super::*;\n" + "".join(chunks), "utf-8")
    return mapping


if __name__ == "__main__":
    assert (ROOT / "build/pdf-structure-before-20261030/manifest.json").exists()
    processing()
    inference()
    binding_items = bindings()
    # Collapse intermediate directory moves into original -> final paths.
    def final(path):
        seen = set()
        while path in MOVES:
            assert path not in seen
            seen.add(path)
            path = MOVES[path]
        return path
    manifest = dict(moves={source: final(target) for source, target in MOVES.items()}, binding_items=binding_items)
    (ROOT / "build/pdf-structure-moves-20261030.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print("Moved files", len(manifest["moves"]), "binding items", len(binding_items))
