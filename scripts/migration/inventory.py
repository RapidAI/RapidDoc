"""不加载模型，通过 AST 清点公开 API、配置与模型资源。"""

import argparse
import ast
import re
import yaml
from common import ROOT, digest, environment, load_manifest, write_json


def inventory():
    """记录源码契约、模型哈希和已知缺口，不将缺失语料标为通过。"""
    source = ROOT / "rapid_doc/main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    contracts = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"RapidDoc", "RapidDocOutput"}:
            contracts[node.name] = {
                item.name: ast.unparse(item.args) for item in node.body
                if isinstance(item, ast.FunctionDef) and (not item.name.startswith("_") or item.name in {"__init__", "__call__", "__iter__"})}
            contracts[node.name]["fields"] = [ast.unparse(item) for item in node.body if isinstance(item, ast.AnnAssign)]
    enum_tree = ast.parse((ROOT / "rapid_doc/utils/enum_class.py").read_text(encoding="utf-8"))
    contracts["MakeMode"] = [ast.unparse(node) for node in enum_tree.body if isinstance(node, ast.ClassDef) and node.name == "MakeMode"]
    resources = []
    model_catalog = []
    for path in sorted((ROOT / "rapid_doc").rglob("*")):
        if path.is_file() and path.suffix in {".onnx", ".yaml", ".yml"}:
            resources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes()), "size": path.stat().st_size})
            if path.name == "default_models.yaml":
                for name, config in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items():
                    model_catalog.append({"name": name, "source": path.relative_to(ROOT).as_posix(),
                                          "declared_config": config,
                                          "scope": "excluded" if "unitable" in name.lower() else "parity-candidate",
                                          "validation": "Not loaded; license and tensor contract pending"})
    upstream = (ROOT / "rapid_doc/_vendor/docvortex_core/UPSTREAM.md").read_text(encoding="utf-8")
    missing = sorted({name for name in re.findall(r"`((?:docs|tests)/[^`]+)`", upstream) if not (ROOT / name).exists()})
    if not (ROOT / "tests/fixtures/docvortex_migration").exists():
        missing.append("tests/fixtures/docvortex_migration")
    return {"environment": environment(), "api": contracts, "api_source_sha256": digest(source.read_bytes()),
            "resources": resources, "model_catalog": model_catalog, "missing_references": missing,
            "scope": {"parity": ["Python API (captured signatures)", "native PDF", "six Office formats", "ONNX CPU/CUDA"],
                      "excluded": ["VLM", "UniTable torch", "PyTorch runtime", "OpenVINO backend", "Paddle backend"],
                      "deferred": ["CANN", "DirectML", "Node.js", "Java", "Go", "C ABI"]},
            "model_inventory_scope": "Repository only; external caches and tensor contracts pending"}


def main():
    """--check 严格检查登记语料；--require-complete 还要求历史引用恢复。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", default="build/migration/inventory.json")
    args = parser.parse_args()
    result = inventory()
    if args.check:
        load_manifest(ROOT / "tests/migration/corpus_manifest.json")
    write_json(args.output, result)
    print(f"Inventory: {args.output}; missing historical references: {len(result['missing_references'])}")
    if args.require_complete and result["missing_references"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
