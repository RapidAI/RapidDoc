"""Focused original-Python collection references without historical build inputs."""
import ast
import copy
from dataclasses import fields
import gzip
import hashlib
import json

from common import ROOT
from export_native_inline_clusters import clean
from export_native_spacing import reference as spacing_reference


def export_prefix(name, stop=None, capture=False):
    path = ROOT / 'scripts/migration' / name
    tree = ast.parse(path.read_text(encoding='utf8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    if stop:
        index = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == stop for t in n.targets))
        function.body = function.body[:index] + [ast.Return(value=ast.Name(id='cases', ctx=ast.Load()))]
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = node.value.replace('build/migration/', 'build/native-collect-borrow-references-20261101/')
    tree.body = [n for n in tree.body if not isinstance(n, ast.If)]
    namespace = {'__name__': '_collection_reference'}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    captured = []
    if capture:
        namespace['write_json'] = lambda target, value: captured.append(value)
    result = namespace['main']()
    return captured[0]['cases'] if capture else result


def main():
    target = ROOT / 'tests/fixtures/native_collection_reference.json.gz'
    if target.exists():
        raise FileExistsError(target)
    from loguru import logger
    logger.remove()
    evidence = export_prefix('export_native_inline_evidence.py', stop='source')
    glyphs = export_prefix('export_native_glyph_weight.py', capture=True)
    for c in glyphs:
        c['png_bytes'] = list((ROOT / c['png']).read_bytes()) if c['png'] else None
    old = json.loads(gzip.decompress((ROOT / 'tests/fixtures/native_owned_line_reference.json.gz').read_bytes()))
    spacing = []
    for c in old['notes'][:16]:
        for before in [None, [], [1, 3, 5]]:
            q = dict(lines=c['request']['lines'], space_before=before)
            spacing.append(dict(label=c['label'] + '/' + str(before), request=q, expected=spacing_reference(q, 'prepare')))
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox
    names = [f.name for f in fields(_LineItem) if not f.name.startswith('_')]
    lines = []
    for c in old['scripts']:
        for rotation in [0, 90, 180, 270]:
            q = dict(chars=copy.deepcopy(c['request']['all_chars']), page_size=c['request']['page_size'],
                     page_rotation=rotation, supported_angles=[0., 90., 270.])
            chars = copy.deepcopy(q['chars'])
            for char in chars:
                char['bbox'] = Bbox(char['bbox'])
            result = _build_native_line_items_from_chars(chars, tuple(q['page_size']), page_rotation=rotation,
                                                        supported_angles=q['supported_angles'])
            lines.append(dict(label=c['label'] + '/' + str(rotation), request=q,
                              expected=[{k: clean(getattr(l, k)) for k in names} for l in result]))
    paths = [ROOT / 'scripts/migration' / p for p in ['export_native_inline_evidence.py', 'export_native_glyph_weight.py']]
    paths += [ROOT / 'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf' / p for p in
              ['native_text.py', 'inline/detection.py', 'inline/spacing.py', 'inline/glyph_weight.py']]
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target.write_bytes(gzip.compress(json.dumps(clean(dict(sources=sources, lines=lines, spacing=spacing,
                                                        evidence=evidence, glyphs=glyphs)), ensure_ascii=False).encode(), mtime=0))
    print(dict(lines=len(lines), spacing=len(spacing), evidence=len(evidence), glyphs=len(glyphs)))


if __name__ == '__main__':
    main()
