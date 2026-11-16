from unittest.mock import Mock

import numpy as np
import pytest

from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.main import (
    PPFormulaNetPlusModelHandler,
)


@pytest.mark.parametrize("retry,accept", [(r"y=\frac{1}{2}", True), (" ", False)])
def test_retry_requires_repetitive_output_and_distributed_speckles(retry, accept):
    clean = np.full((60, 90, 3), 255, dtype=np.uint8)
    clean[18:42, 42:47] = 0
    noisy = clean.copy()
    noisy[::4, ::4] = 100
    repeated = r"\mathrm{d}(" * 80
    handler = PPFormulaNetPlusModelHandler.__new__(PPFormulaNetPlusModelHandler)
    handler._recognize = Mock(side_effect=[[repeated, repeated, "x"], [retry]])

    results = handler([clean, noisy, noisy])

    assert [result.rec_formula for result in results] == [
        repeated, retry if accept else repeated, "x",
    ]
    assert handler._recognize.call_count == 2
    cleaned = handler._recognize.call_args.args[0][0]
    assert np.all(cleaned[::4, 0] == 255)
    np.testing.assert_array_equal(cleaned[18:42, 42:47], noisy[18:42, 42:47])
