"""Compare DOC raw image placement and decoded pixels, including SVG fallbacks.

Run doc_objects Rust tests first to export build/doc-objects-raw.json.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter


def images(value):
    if isinstance(value, dict):
        if value.get('image_base64'):
            yield value['image_base64']
        for child in value.values():
            yield from images(child)
    elif isinstance(value, list):
        for child in value:
            yield from images(child)


def pixels(data):
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert('RGBA')
        return [image.size, hashlib.sha256(image.tobytes()).hexdigest()]


def canonical(uri):
    mime, data = uri.split(',', 1)
    data = base64.b64decode(data)
    if 'svg' not in mime:
        return pixels(data)
    root = ET.fromstring(data)
    for item in root.iter():
        if item.attrib.get('id') == 'metafile-render-raster-fallback':
            item.text = json.dumps(pixels(base64.b64decode(item.text)))
    return ET.tostring(root)


def main():
    converter = DocConverter()
    converter.convert(io.BytesIO((ROOT / 'demo/ms_office_docs/docx_01.doc').read_bytes()))
    rust = json.loads((ROOT / 'build/doc-objects-raw.json').read_text(encoding='utf-8'))
    expected, actual = list(images(converter.pages)), list(images(rust['pages']))
    differences = [i for i, (a, b) in enumerate(zip(expected, actual)) if canonical(a) != canonical(b)]
    report = {'python_references': len(expected), 'rust_references': len(actual),
              'canonical_differences': differences,
              'comparison': 'RGBA pixels; SVG XML including decoded fallback pixels; body image and chart-preview order'}
    (ROOT / 'build/doc-objects-image-comparison-20261101.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    assert len(expected) == len(actual) and not differences, report


if __name__ == '__main__':
    main()
