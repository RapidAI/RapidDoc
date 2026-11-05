"""迁移工具必须能发现缺语料、内容变更和不稳定生成。"""

import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/migration"))
from common import digest, load_manifest, parse_case
from compare_results import differences


@pytest.mark.parametrize("left,right", [
    ({"text": "中文 English"}, {"text": "中文English"}),
    (["first", "second"], ["second", "first"]),
    ({"unknown": 1}, {}),
    ({"value": True}, {"value": 1}),
    ([1], [1, 2]),
])
def test_comparator_preserves_semantics(left, right):
    """未知字段、文字空格、顺序和类型不能被规范化隐藏。"""
    assert differences(left, right)


@pytest.mark.parametrize("failure", ["empty", "missing", "hash", "duplicate"])
def test_manifest_rejects_invalid_inputs(tmp_path, failure):
    """清单缺文件、哈希错误、重复 ID 或无样本均应失败。"""
    manifest = json.loads((ROOT / "tests/migration/corpus_manifest.json").read_text(encoding="utf-8"))
    if failure == "empty":
        manifest["cases"] = []
    elif failure == "missing":
        manifest["cases"][0]["input"] = "tests/migration/does-not-exist.doc"
    elif failure == "hash":
        manifest["cases"][0]["sha256"] = "0" * 64
    else:
        manifest["cases"].append(manifest["cases"][0])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_manifest(path)


def test_restored_fixture_provenance_matches_bytes():
    """恢复样本必须与记录的来源哈希一致，包括字符去重的 JSON 片段。"""
    provenance = json.loads((ROOT / "tests/fixtures/provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["files"]) == 15
    for item in provenance["files"]:
        assert digest((ROOT / item["target"]).read_bytes()) == item["sha256"]


@pytest.mark.parametrize("case", load_manifest(ROOT / "tests/migration/corpus_manifest.json"), ids=lambda case: case["id"])
def test_registered_python_output_matches_golden(case):
    """当前解析器应保留登记样本的完整输出与资产字节。"""
    actual, _ = parse_case(case)
    expected = json.loads((ROOT / "tests/migration/golden" / case["expected"]).read_text(encoding="utf-8"))
    assert not differences(expected, actual)
    if case.get("capture") == "pdf-facts":
        assert actual["pages"]
        if case["id"] == "pdf_scan":
            assert not actual["pages"][0]["chars"]
            assert actual["pages"][0]["image_bboxes"]
        else:
            assert actual["pages"][0]["chars"]
            assert actual["pages"][0]["drawing_lines"]
    else:
        assert actual["markdown"].strip()
