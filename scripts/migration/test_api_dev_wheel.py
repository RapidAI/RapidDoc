"""Run affected SDK tests against an unpacked dev wheel, without replacing the installed release."""
from pathlib import Path
import runpy
import sys
import zipfile

root=Path(__file__).resolve().parents[2]
wheel=root/'build/dev-wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl'
directory=root/'build/api-dev-sdk';directory.mkdir(parents=True,exist_ok=True)
# These are generated files from the previous development wheel, not sources.
for retired in ['_classification_rules.py','_classification_probe.py','_classifier_bridge.py','_layout_rules.py']:
    (directory/'rapid_doc_rust'/retired).unlink(missing_ok=True)
with zipfile.ZipFile(wheel)as package:package.extractall(directory)
sys.path.insert(0,str(directory))
# Run serially: Windows keeps the extension DLL mapped until process exit.
test=sys.argv[1];sys.argv=[test,str(root),*sys.argv[2:]]
runpy.run_path(str(root/'bindings/python/tests'/test),run_name='__main__')
