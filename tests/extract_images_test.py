# SPDX-FileCopyrightText: 2025 geisserml <geisserml@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause

# TODO test-cover use_bitmap, format and render
# https://github.com/pypdfium2-team/pypdfium2/issues/137
# https://github.com/pypdfium2-team/pypdfium2/blob/main/src/pypdfium2/_cli/extract_images.py

import traceback
from pathlib import Path
import pypdfium2.raw as pdfium_c
import pypdfium2._helpers as pdfium


def extract_images_with_bbox(
    pdf_path: str,
    output_dir: str,
    pages: list[int] | None = None,
    max_depth: int = 15,
    use_bitmap: bool = False,
    img_format: str | None = None,
    render: bool = False,
    scale_to_original: bool = True,
):
    """
    提取 PDF 中图片，同时获取每张图片的 bbox。
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    if use_bitmap and not img_format:
        img_format = "png"

    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)
    if pages is None:
        pages = list(range(total_pages))

    n_pdigits = len(str(total_pages))
    results = []

    for i in pages:
        page = pdf[i]
        images = list(
            page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=max_depth)
        )
        n_idigits = len(str(len(images)))

        for j, image in enumerate(images):
            tag = f"{i + 1:0{n_pdigits}}_{j + 1:0{n_idigits}}"
            prefix = output_dir / f"{Path(pdf_path).stem}_{tag}"

            # === 获取 bbox ===
            try:
                x1, y1, x2, y2 = image.get_pos() # PDF 左下角原点坐标
                page_width, page_height = page.get_size()

                width = abs(x2 - x1)
                height = abs(y2 - y1)
                # 过滤掉“点状”小图像
                MIN_IMAGE_WIDTH = 5
                MIN_IMAGE_HEIGHT = 5
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    image.close()
                    continue

                # 转换为左上角原点坐标
                new_x1 = x1
                new_x2 = x2
                new_y1 = page_height - y2
                new_y2 = page_height - y1

                bbox = (new_x1, new_y1, new_x2, new_y2)
            except Exception:
                bbox = None

            try:
                if use_bitmap:
                    # ❶ 检查是否支持 scale_to_original 参数
                    from inspect import signature
                    sig = signature(image.get_bitmap)
                    kwargs = {}
                    if "render" in sig.parameters:
                        kwargs["render"] = render
                    if "scale_to_original" in sig.parameters:
                        kwargs["scale_to_original"] = scale_to_original

                    pil_image = image.get_bitmap(**kwargs).to_pil()
                    pil_image.save(f"{prefix}.{img_format}")
                # else:
                #     image.extract(prefix, fb_format=img_format)
            except pdfium.PdfiumError:
                traceback.print_exc()
            finally:
                image.close()

            results.append(
                {
                    "page_index": i,
                    "image_index": j,
                    "file": f"{prefix}.{img_format or 'bin'}",
                    "bbox": bbox,
                }
            )

        page.close()

    pdf.close()
    print("✅ 提取完成！输出目录：", output_dir)
    return results


if __name__ == "__main__":
    pdf_path =  "D:\\file\\text-pdf\\示例1-论文模板.pdf"
    output_dir = "./output_images"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result = extract_images_with_bbox(
        pdf_path=pdf_path,
        output_dir=output_dir,
        use_bitmap=True,
        render=True,
        scale_to_original=True,
        img_format="png",
    )

    for r in result:
        print(
            f"Page {r['page_index'] + 1}, Image {r['image_index'] + 1}, "
            f"File: {r['file']}, BBox: {r['bbox']}"
        )
