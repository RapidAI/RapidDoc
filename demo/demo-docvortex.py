"""使用 DocVortex 公开接口完成原生解析、Markdown 导出和结果包保存。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from docvortex import parse


def run_demo(input_path: Path, output_dir: Path, *, page_range: str = "", overwrite: bool = False) -> None:
    """解析一次后复用结果导出，结果包可在没有源文件的环境中恢复。"""
    start_time = time.time()
    result = parse(input_path, page_range=page_range, keep_model_json=True)
    markdown = result.export(output_dir / f"{input_path.stem}.md", overwrite=overwrite)
    bundle = result.save_bundle(output_dir / f"{input_path.stem}.docvortex", overwrite=overwrite)
    print(f"Markdown: {markdown.path}")
    print(f"Bundle: {bundle.path}")
    print(f"total cost: {time.time() - start_time:.2f}s")


def main() -> None:
    """运行默认文本 PDF 示例，也允许传入其他支持的原生文档。"""
    start_time = time.time()
    demo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=demo_dir / "pdfs/示例1-论文模板.pdf")
    # parser.add_argument("input", nargs="?", type=Path, default= r"D:\file\text-pdf\佳能打印机.pdf")
    parser.add_argument("--output-dir", type=Path, default=demo_dir / "output2")
    parser.add_argument("--page-range", default="", help="PDF 页码范围，例如 1-5、r1 或 all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_demo(args.input, args.output_dir, page_range=args.page_range, overwrite=args.overwrite)
    print(f"total cost: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
