"""迁移基线共用的严格清单、序列化和解析入口。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(data):
    """计算输入字节的 SHA256。"""
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    """以稳定键序保存 UTF-8 JSON，不修改数组、文本或未知字段。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(path):
    """校验清单、唯一 ID、输入存在性和哈希；空清单直接失败。"""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if not cases:
        raise ValueError("Corpus must contain at least one case")
    seen = set()
    for case in cases:
        for key in ("id", "input", "sha256", "format", "license", "tags", "config", "model_manifest", "expected"):
            if key not in case:
                raise ValueError(f"Missing manifest field: {key}")
        if not re.fullmatch(r"[a-z0-9_-]+", case["id"]) or case["id"] in seen:
            raise ValueError(f"Invalid or duplicate case ID: {case['id']}")
        seen.add(case["id"])
        source = (ROOT / case["input"]).resolve()
        source.relative_to(ROOT)
        if digest(source.read_bytes()) != case["sha256"]:
            raise ValueError(f"Input hash mismatch: {case['id']}")
        if case["expected"] != case["id"] + ".json":
            raise ValueError("Expected output must match case ID")
    return cases


def environment():
    """记录源码状态及已安装运行库，不读取凭据或完整环境变量。"""
    packages = {}
    for name in ("pypdfium2", "onnxruntime", "onnxruntime-gpu", "numpy", "python-docx", "python-pptx", "openpyxl", "reportlab"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)}


def parse_case(case, engine="python"):
    """解析单个样本，分开计量读取、引擎构造和解析；不宣称内部阶段耗时。"""
    if engine != "python":
        raise ValueError("Rust engine is not implemented yet")
    from rapid_doc import RapidDoc

    start = perf_counter()
    data = (ROOT / case["input"]).read_bytes()
    read_end = perf_counter()
    parser = RapidDoc(**case["config"]) if case.get("capture") != "pdf-facts" else None
    init_end = perf_counter()
    if case.get("capture") == "pdf-facts":
        from pdf_corpus import extract_facts
        value = extract_facts(data)
        end = perf_counter()
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)), {
            "read_s": read_end - start, "init_s": init_end - read_end,
            "parse_s": end - init_end, "total_s": end - start}
    result = parser(data)
    end = perf_counter()
    import base64
    value = {"markdown": result.markdown, "middle_json": result.middle_json,
             "content_list_json": result.content_list_json,
             "assets": {name: {"sha256": digest(blob), "size": len(blob),
                                "base64": base64.b64encode(blob).decode("ascii")}
                        for name, blob in result.images.items()}}
    # 输出比较以 JSON 契约为边界，字符串枚举按标准 JSON 编码还原为字符串。
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return value, {"read_s": read_end - start, "init_s": init_end - read_end,
                   "parse_s": end - init_end, "total_s": end - start}
