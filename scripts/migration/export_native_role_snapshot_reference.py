"""Freeze the independent Python alias/tie scenes before historical PDF inputs."""
import ast
import gzip
import hashlib
import json

from common import ROOT


def main():
    target = ROOT / 'tests/fixtures/native_role_snapshot_reference.json.gz'
    if target.exists():
        raise FileExistsError(target)
    script = ROOT / 'scripts/migration/export_native_text_role_aliases.py'
    tree = ast.parse(script.read_text(encoding='utf8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    for i, node in enumerate(function.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'f' for t in node.targets):
            function.body = function.body[:i] + [ast.Return(value=ast.Name(id='cases', ctx=ast.Load()))]
            break
    else:
        raise AssertionError('Historical geometry fixture boundary changed')
    tree.body = [n for n in tree.body if not isinstance(n, ast.If)]
    namespace = {'__name__': '_alias_snapshot_export'}
    exec(compile(ast.fix_missing_locations(tree), str(script), 'exec'), namespace)
    cases = namespace['main']()
    assert len(cases) == 3
    source_paths = [script, ROOT / 'scripts/migration/export_native_text_roles.py']
    source_paths += [ROOT / 'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf' / p for p in [
        'pipeline.py', 'title_analysis/native_boundaries.py', 'title_analysis/structural.py',
        'text_assembly/continuity.py', 'native_cjk_bands.py']]
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    target.write_bytes(gzip.compress(json.dumps(dict(sources=sources, cases=cases), ensure_ascii=False).encode(), mtime=0))
    print(len(cases), 'original Python alias/tie scenes')


if __name__ == '__main__':
    main()
