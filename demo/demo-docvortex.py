"""使用 RapidDoc 内置 DocVortex 核心解析 PDF 与 Office 文件。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rapid_doc import RapidDoc


def run_demo(input_path: Path, output_dir: Path, *, start_page: int = 0, end_page: int | None = None) -> None:
    """生成 RapidDoc 的 Markdown、图片和 JSON 输出。"""
    start_time = time.time()
    parser = RapidDoc(layout_config={"model_type": "docvortex_rule"}, formula_enable=False)
    result = parser(input_path, output_dir=output_dir, start_page_id=start_page, end_page_id=end_page)
    print(f"Output directory: {output_dir}")
    print(f"Markdown characters: {len(result.markdown)}")
    print(f"total cost: {time.time() - start_time:.2f}s")


def main() -> None:
    """运行默认文本 PDF 示例，也允许传入其他支持的原生文档。"""
    start_time = time.time()
    demo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=demo_dir / "pdfs/1.pdf")
    parser.add_argument("--output-dir", type=Path, default=demo_dir / "output2")
    parser.add_argument("--start-page", type=int, default=0, help="PDF 起始页，0 为第一页")
    parser.add_argument("--end-page", type=int, default=None, help="PDF 结束页")
    args = parser.parse_args()
    run_demo(args.input, args.output_dir, start_page=args.start_page, end_page=args.end_page)
    print(f"total cost: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
