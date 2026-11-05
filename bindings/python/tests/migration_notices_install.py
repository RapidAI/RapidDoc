"""Check attribution in the installed wheel without loading parser models."""

import hashlib
import json
from importlib.resources import files

root = files("rapid_doc_rust").joinpath("licenses")
bundle = root.joinpath("MIGRATION-NOTICES.txt").read_bytes()
index = json.loads(root.joinpath("migration-notices.json").read_text(encoding="utf8"))
assert len(index["entries"]) == 62
assert sum(len(e["original_paths"]) for e in index["entries"]) == 120
for entry in index["entries"]:
    assert hashlib.sha256(bundle[entry["start"] : entry["end"]]).hexdigest() == entry["sha256"]
    assert not root.joinpath(entry["name"]).is_file(), entry["name"]
for name in ["PDFTEXT.txt", "PYPDFIUM2.txt", "PYTHON-CODECS-LICENSE.txt", "NUMPY-PAIRWISE.txt"]:
    assert root.joinpath(name).is_file(), name
print(
    "Installed attribution: 62 verbatim sections, 120 original paths, original license texts retained"
)
