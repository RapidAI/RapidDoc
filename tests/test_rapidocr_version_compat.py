import ast
from pathlib import Path

import pytest
from packaging.version import Version

from rapid_doc.model.ocr.onnx_ops_session import configure_cuda_onnx


@pytest.mark.parametrize('name', ['torch_backend.py', 'ocr_patch.py', 'rapid_ocr.py'])
@pytest.mark.parametrize('installed', ['3.4.0', '3.4.2', '3.4.3', '3.8.0', '3.9.2', '3.10.0', '3.10.0rc1'])
def test_compatibility_branches_use_numeric_versions(name, installed):
    # Execute the real module's version assignment/conditions without loading
    # optional inference engines or downloading models for every old release.
    path = Path(__file__).parents[1] / 'rapid_doc/model/ocr' / name
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'rapidocr_version' for t in node.targets))
    env = {'version': lambda _: installed, 'Version': Version}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), 'exec'), env)
    checks = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
              and isinstance(node.left, ast.Name) and node.left.id == 'rapidocr_version']
    assert checks
    for check in checks:
        threshold = next(node.value for node in ast.walk(check.comparators[0])
                         if isinstance(node, ast.Constant) and isinstance(node.value, str))
        expected = (Version(installed) >= Version(threshold) if isinstance(check.ops[0], ast.GtE)
                    else Version(installed) <= Version(threshold))
        assert eval(compile(ast.Expression(check), str(path), 'eval'), env) is expected


@pytest.mark.parametrize('installed,field', [('3.4.2', 'gpu_id'), ('3.4.3', 'cuda_ep_cfg.device_id'),
                                           ('3.10.0', 'cuda_ep_cfg.device_id'),
                                           ('3.10.0rc1', 'cuda_ep_cfg.device_id')])
def test_cuda_config_handles_releases_and_prereleases(installed, field):
    params = {'Det.engine_type': 'onnxruntime', 'Rec.engine_type': 'onnxruntime'}
    configure_cuda_onnx(params, 'cuda:2', installed)
    assert params[f'EngineConfig.onnxruntime.{field}'] == 2
