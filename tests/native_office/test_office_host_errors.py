from io import BytesIO

import pytest
from _legacy_ppt_test_utils import _build_cfb
from rapid_doc import RapidDoc
from rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes, guess_suffix_by_path
from rapid_doc._vendor.docvortex_core.analyzers.native.office.errors import LegacyOfficeEncryptedError


@pytest.mark.parametrize("suffix", ["docx", "xlsx", "pptx"])
def test_encrypted_modern_office_has_same_error_for_all_inputs(tmp_path, suffix):
    data = _build_cfb([("EncryptionInfo", b"\0" * 64), ("EncryptedPackage", b"\0" * 64)])
    path = tmp_path / ("encrypted." + suffix)
    path.write_bytes(data)
    for call in [lambda: guess_suffix_by_bytes(data), lambda: guess_suffix_by_path(path),
                 lambda: RapidDoc()(data), lambda: RapidDoc()(path), lambda: RapidDoc()(BytesIO(data))]:
        with pytest.raises(LegacyOfficeEncryptedError) as error:
            call()
        assert error.value.code == "encrypted"
