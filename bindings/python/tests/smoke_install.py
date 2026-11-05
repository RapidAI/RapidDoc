"""在干净环境中直接执行，确认 wheel 无原 Python 包依赖。"""

import importlib.util
from rapid_doc_rust import capabilities, resolve_page_range, validate_rect

assert importlib.util.find_spec("rapid_doc") is None
assert importlib.util.find_spec("torch") is None
assert validate_rect([0, 0, 20, 30]) == [0.0, 0.0, 20.0, 30.0]
assert resolve_page_range(2, None, 5) == (2, 5)
for value in ([2, 0, 1, 1], [0, 0, float("nan"), 1]):
    try:
        validate_rect(value)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid geometry accepted")
assert capabilities()["formats"] == ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"]
print("Installed native wheel: contracts pass, no rapid_doc or torch")
