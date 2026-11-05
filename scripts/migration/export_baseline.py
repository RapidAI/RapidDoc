"""导出当前 Python 输出；默认拒绝覆盖已有 golden。"""

import argparse
from pathlib import Path
from common import environment, load_manifest, parse_case, write_json


def main():
    """先验证所有输入，再导出完整资产和输出，错误不转换为空结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = load_manifest(args.manifest)
    targets = [args.output / case["expected"] for case in cases]
    if any(path.exists() for path in targets) or (args.output / "environment.json").exists():
        parser.error("Output already exists; use a new directory and review differences")
    outputs = [(case, *parse_case(case)) for case in cases]
    for case, value, timings in outputs:
        write_json(args.output / case["expected"], value)
        print(case["id"], timings)
    write_json(args.output / "environment.json", environment())


if __name__ == "__main__":
    main()
