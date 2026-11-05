"""Freeze original spreadsheet regions, gap scoring, table IR and HTML projection.

Openpyxl is a reference-data producer only; native source types consume no Python
objects. Full XLSX package/media reading is verified in its own later phase.
"""
import copy,gzip,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from openpyxl import Workbook,load_workbook
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.spreadsheet.projector import SpreadsheetProjector
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.spreadsheet.region_discovery import summarize_connected_regions,gap_candidate_penalty
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.spreadsheet.html import render_spreadsheet_table
    from loguru import logger
    logger.remove();cases=[]
    def capture(name,sheet,media=None,math=None):
        projector=SpreadsheetProjector();projector.workbook=sheet.parent;projector.math_map=math or{};projector.table_image_map=media or{}
        semantic,span=projector._sheet_semantic_predicates(sheet)
        from openpyxl.cell.cell import Cell
        default_cell=dict(row=0,col=0,text='',row_span=1,col_span=1,styles=projector._extract_cell_style(Cell(sheet)),media=[],equations=[],text_is_html=False,source_row=None,source_col=None)
        cells=[]
        # Evaluate before materialization fills rectangle holes in _cells.
        positions=set(sheet._cells)|{(r+1,c+1)for r,c in list((media or{}))+list((math or{}))}
        for r,c in sorted(positions):
            cell=sheet.cell(r,c)
            built=projector._build_excel_cell(sheet,0,0,r-1,c-1)
            cells.append(dict(row=r-1,col=c-1,has_value=cell.value is not None,semantic=semantic(r-1,c-1),cell=built.model_dump()))
        merges=[dict(min_row=m.min_row-1,max_row=m.max_row-1,min_col=m.min_col-1,max_col=m.max_col-1)for m in sheet.merged_cells.ranges]
        bounds=projector._find_true_data_bounds(sheet);queries=[]
        for gap in [0,1,2,3,None]:
            regions=projector._discover_sheet_regions(sheet,bounds,gap)if gap is not None else None
            if gap is None:selected,penalty,tables=projector._select_best_gap_candidate(sheet)
            else:selected=gap;penalty=0.;tables=projector._find_data_tables_with_gap(sheet,gap)
            q=dict(gap=gap,selected=selected,penalty=penalty,tables=[t.model_dump()for t in tables],html=[render_spreadsheet_table(t)for t in tables],blocks=[projector._build_block_from_excel_table(t)for t in tables])
            if regions is not None:
                q['regions']=[dict(cells=sorted(r.cells),row_start=r.row_start,row_end=r.row_end,col_start=r.col_start,col_end=r.col_end)for r in regions];q['summary']=summarize_connected_regions(regions,semantic,span);q['score']=gap_candidate_penalty(q['summary'])
            queries.append(q)
        rows=sorted({r for r,c in sheet._cells if sheet._cells[(r,c)].value is not None})[:4];cols=sorted({c for r,c in sheet._cells if sheet._cells[(r,c)].value is not None})[:4]
        synthetic=None
        if rows and cols:
            table=projector._build_synthetic_table_from_sheet_selection(sheet,[r-1 for r in rows],[c-1 for c in cols]);synthetic=dict(rows=[r-1 for r in rows],cols=[c-1 for c in cols],table=table.model_dump(),html=render_spreadsheet_table(table))
        cases.append(dict(name=name,cells=cells,merges=merges,default_cell=default_cell,bounds=[bounds.min_row-1,bounds.max_row-1,bounds.min_col-1,bounds.max_col-1],queries=queries,synthetic=synthetic))
    rng=random.Random(31102)
    for index in range(55):
        wb=Workbook();sheet=wb.active
        for row in range(1,rng.randrange(2,13)):
            for col in range(1,rng.randrange(2,10)):
                if rng.random()<.35:sheet.cell(row,col,rng.choice(['A','B','  ',0,False,'<tag>\n & x','DISPIMG unknown']))
        if index%3==0:sheet.merge_cells(start_row=1,start_column=1,end_row=2,end_column=3)
        if index%4==0:
            font=copy.copy(sheet.cell(1,1).font);font.b=True;sheet.cell(30,25).font=font
        capture('generated '+str(index),sheet)
    from openpyxl.cell.rich_text import CellRichText,TextBlock
    from openpyxl.cell.text import InlineFont
    for index in range(8):
        wb=Workbook();sheet=wb.active;sheet['A1']='title';sheet['A2']='value';sheet['A3']=CellRichText(['plain ',TextBlock(InlineFont(b=True,u='single'),'bold space')]);sheet['B2']='hyperlink';sheet['B2'].hyperlink='https://example.com/a?b=1&c=2';sheet['C4']='DISPIMG missing';sheet.merge_cells('A5:C6')
        media={(1,0):['<img src="data:image/png;base64,AA==" />']}if index%2 else{};math={(4,0):['x+y']}if index%3 else{};capture('media merge rich '+str(index),sheet,media,math)
    for path in sorted((ROOT/'demo/xlsx').glob('*.xlsx'))+[ROOT/'demo/ms_office_docs/xlsx_01.xlsx']:
        wb=load_workbook(path,data_only=True,rich_text=True)
        for sheet in wb.worksheets:capture('real '+path.name+' '+sheet.title,sheet)
        wb.close()
    target=ROOT/'tests/fixtures/office_rust_reference_068/spreadsheet_projection.json.gz';target.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True,default=str).encode(),mtime=0));print('Spreadsheet projection:',len(cases),'sheets;',sum(len(c['queries'])for c in cases),'gap queries')
if __name__=='__main__':main()
