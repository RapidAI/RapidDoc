"""Native terminal progress: English stages, redirection, opt-out and throttling."""

import os
from pathlib import Path
import subprocess
import sys
import rapid_doc_rust

script = """
from rapid_doc_rust._native import ParseProgress
p=ParseProgress('demo1',13)
assert p.update('表格识别',5)=='table'
for i in range(1000):p.update('页面组装与图片资产',5)
p.clear()
"""
for mode in ["auto", "off", "always"]:
    process = subprocess.run(
        [sys.executable, "-I", "-c", script],
        env=dict(os.environ, RAPIDDOC_PROGRESS=mode),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert process.returncode == 0, process.stderr
    if mode == "always":
        assert "6/13 table" in process.stderr, process.stderr
        assert len(process.stderr) < 1000, "Progress output was not throttled"
    else:
        assert not process.stderr, process.stderr
assert not (Path(rapid_doc_rust.__file__).parent / "_polygon_crops.py").exists()
print(
    "Native progress English phases, redirected output, off/always controls and throttling passed"
)
