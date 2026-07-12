from rapid_doc.utils.table_merge import (
    _apply_cell_merge,
    _infer_sparse_continuation_cell_merge,
    build_table_state_from_html,
)


def test_sparse_page_break_row_continues_previous_cells():
    previous = build_table_state_from_html(
        '<table><tr><td>9</td><td>海南（取送</td><td>1</td>'
        '<td>山东核电设备制造有</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td></td><td>费）</td><td></td><td>限公司</td></tr>'
        '<tr><td>10</td><td>海南（不含税）</td><td>1</td><td>山东公司</td></tr></table>'
    )

    flags = _infer_sparse_continuation_cell_merge(previous, current, 0)
    assert flags == [0, 1, 0, 1]
    current.owner_block['cell_merge'] = flags
    _apply_cell_merge(previous, current, 0)

    assert len(current.rows) == 1
    assert previous.rows[-1].find_all('td')[1].get_text() == '海南（取送费）'
    assert previous.rows[-1].find_all('td')[3].get_text() == '山东核电设备制造有限公司'


def test_sparse_first_row_with_key_is_not_inferred_as_continuation():
    previous = build_table_state_from_html(
        '<table><tr><td>9</td><td>old</td><td>1</td></tr></table>'
    )
    current = build_table_state_from_html(
        '<table><tr><td>10</td><td>new</td><td></td></tr>'
        '<tr><td>11</td><td>next</td><td>1</td></tr></table>'
    )
    assert _infer_sparse_continuation_cell_merge(previous, current, 0) is None
