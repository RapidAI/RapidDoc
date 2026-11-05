"""严格比较输出：不折叠空格、不调整块顺序、不删除未知字段。"""

import argparse
import json
from pathlib import Path
from common import load_manifest


def differences(expected, actual, path="$" ):
    """递归报告 JSON 路径，类型、缺字段和列表长度差异均保留。"""
    if type(expected) is not type(actual):
        return [path + ": type mismatch"]
    if isinstance(expected, dict):
        result = [f"{path}.{key}: missing/extra" for key in sorted(expected.keys() ^ actual.keys())]
        for key in sorted(expected.keys() & actual.keys()):
            result.extend(differences(expected[key], actual[key], f"{path}.{key}"))
        return result
    if isinstance(expected, list):
        result = [path + ": length mismatch"] if len(expected) != len(actual) else []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(differences(left, right, f"{path}[{index}]"))
        return result
    return [] if expected == actual else [path + ": value mismatch"]


def main():
    """比较所有登记样本，缺失文件或语义差异返回非零退出码。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for case in load_manifest(args.manifest):
        left = json.loads((args.expected / case["expected"]).read_text(encoding="utf-8"))
        right = json.loads((args.actual / case["expected"]).read_text(encoding="utf-8"))
        failures.extend(f"{case['id']}: {item}" for item in differences(left, right))
    print("\n".join(failures) if failures else "All registered outputs match exactly")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
