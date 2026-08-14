# Copyright (c) Opendatalab. All rights reserved.
from typing import BinaryIO

from rapid_doc.model.xlsx.xlsx_converter import XlsxConverter


def convert_path(file_path: str):
    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO, gap_tolerance: int | None = None):
    converter = XlsxConverter(gap_tolerance=gap_tolerance)
    converter.convert(file_binary)
    return converter.pages

if __name__ == "__main__":
    print(convert_path("test_xlsx/xlsx_01.xlsx"))
