from rapid_doc.model.table.rule_table import build_rule_table, build_track_table


def _page(rows=3, cols=3, missing=None):
    missing = missing or set()
    xs = [10 + 50*i for i in range(cols+1)]
    ys = [20 + 30*i for i in range(rows+1)]
    lines = []
    for r,y in enumerate(ys):
        if ('h',r) not in missing:
            lines.append({'x1':xs[0], 'y1':y, 'x2':xs[-1], 'y2':y})
    for c,x in enumerate(xs):
        if ('v',c) not in missing:
            lines.append({'x1':x, 'y1':ys[0], 'x2':x, 'y2':ys[-1]})
    chars=[]
    for r in range(rows):
        for c in range(cols):
            text=f'{r}{c}'
            x,y=xs[c]+5,ys[r]+8
            for i,ch in enumerate(text):
                chars.append({'char':ch, 'bbox':[x+i*5,y,x+i*5+4,y+8]})
    return {'vector_lines':lines, 'chars':chars}


def test_dense_ruled_grid_builds_native_html():
    result=build_rule_table(_page(), [10,20,160,110])
    assert result is not None
    assert result.rows == 3 and result.columns == 3
    assert result.score >= .99
    assert '<td>00</td>' in result.html
    assert '<td>22</td>' in result.html


def test_missing_outer_rule_is_rejected():
    assert build_rule_table(_page(missing={('v', 3)}), [10,20,160,110]) is None


def test_sparse_decorative_grid_is_rejected():
    page=_page()
    page['chars']=page['chars'][:2]
    assert build_rule_table(page, [10,20,160,110]) is None


def test_html_text_is_escaped():
    page=_page(rows=2, cols=2)
    page['chars'][0]['char']='<'
    result=build_rule_table(page, [10,20,110,80])
    assert result and '&lt;' in result.html


def test_pdf_font_control_hyphen_is_normalized():
    page = _page(rows=2, cols=2)
    page['chars'][0]['char'] = '\u0002'
    result = build_rule_table(page, [10, 20, 110, 80])
    assert result is not None
    assert '\u0002' not in result.html
    assert '<td>-0</td>' in result.html


def test_missing_local_divider_emits_colspan():
    page=_page()
    # Replace the first internal vertical rule with a segment that starts at
    # row 2: the first-row cells are therefore one logical merged cell.
    page['vector_lines']=[line for line in page['vector_lines'] if not (
        line['x1'] == line['x2'] == 60
    )]
    page['vector_lines'].append({'x1':60, 'y1':50, 'x2':60, 'y2':110})
    result=build_rule_table(page, [10,20,160,110])
    assert result is not None
    assert '<td colspan="2">00 01</td>' in result.html


def test_cell_edges_split_into_contiguous_pdf_segments_remain_boundaries():
    page = _page(rows=3, cols=3)
    split_lines = []
    for line in page['vector_lines']:
        if line['x1'] == line['x2']:
            mid = (line['y1'] + line['y2']) / 2
            split_lines.extend([
                {**line, 'y2': mid},
                {**line, 'y1': mid},
            ])
        else:
            mid = (line['x1'] + line['x2']) / 2
            split_lines.extend([
                {**line, 'x2': mid},
                {**line, 'x1': mid},
            ])
    page['vector_lines'] = split_lines

    result = build_rule_table(page, [10, 20, 160, 110])
    assert result is not None
    assert 'rowspan=' not in result.html
    assert 'colspan=' not in result.html
    assert result.html.count('<td>') == 9


def test_vertical_tracks_without_row_rules_use_text_baselines():
    xs=[10,60,110,160]
    lines=[{'x1':x,'y1':20,'x2':x,'y2':140} for x in xs]
    chars=[]
    values=[['Name','Jan','Feb'],['A','1','2'],['B','3','4'],['C','5','6'],['D','7','8']]
    for r,row in enumerate(values):
        for c,text in enumerate(row):
            for i,ch in enumerate(text):
                chars.append({'char':ch,'bbox':[xs[c]+5+i*5,28+r*20,xs[c]+9+i*5,36+r*20]})
    result=build_track_table({'vector_lines':lines,'chars':chars},[10,20,160,140])
    assert result is not None
    assert result.rows==5 and result.columns==3
    assert '<th>Name</th>' in result.html and '<td>8</td>' in result.html


def test_zero_width_shifted_space_does_not_split_number():
    page=_page(rows=2,cols=2)
    page['chars'].append({'char':' ','bbox':[15,31,15,32]})
    result=build_rule_table(page,[10,20,110,80])
    assert result is not None
    assert '>00<' in result.html


def test_rotated_watermark_is_not_assigned_to_cells():
    page = _page(rows=2, cols=2)
    page['chars'].extend([
        {'char': ch, 'bbox': [20 + i * 5, 35, 24 + i * 5, 43], 'rotation': 5.498}
        for i, ch in enumerate('WATERMARK')
    ])
    result = build_rule_table(page, [10, 20, 110, 80])
    assert result is not None
    assert 'WATERMARK' not in result.html
    assert '<td>00</td>' in result.html


def test_track_table_ignores_zero_width_shifted_space():
    xs=[10,60,110,160]
    lines=[{'x1':x,'y1':20,'x2':x,'y2':140} for x in xs]
    chars=[]
    for r,row in enumerate([['Name','Jan','Feb'],['A','636,903','2','3'],['B','4','5'],['C','6','7'],['D','8','9']]):
        for c,text in enumerate(row[:3]):
            for i,ch in enumerate(text): chars.append({'char':ch,'bbox':[xs[c]+5+i*4,28+r*20,xs[c]+8+i*4,36+r*20]})
    chars.append({'char':' ','bbox':[65,51,65,52]})
    result=build_track_table({'vector_lines':lines,'chars':chars},[10,20,160,140])
    assert result is not None
    assert '>636,903<' in result.html


def test_track_table_rejects_repeated_text_outside_open_edges():
    xs=[60,110,160]
    lines=[{'x1':x,'y1':20,'x2':x,'y2':140} for x in xs]
    chars=[]
    for r in range(5):
        for x,text in [(20,str(r)),(70,f'A{r}'),(120,f'B{r}'),(180,f'+{r}')]:
            for i,ch in enumerate(text):
                chars.append({'char':ch,'bbox':[x+i*5,28+r*20,x+i*5+4,36+r*20]})
    assert build_track_table({'vector_lines':lines,'chars':chars},[10,20,210,140]) is None
