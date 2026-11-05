"""Generate historical Python reference copies under build; never overwrite the SDK."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'build/migration/python-compat-reference'

def source(path):return (ROOT/path).read_text(encoding='utf-8')
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    records={}
    for src,dst in [
        ('rapid_doc/utils/draw_bbox.py','_drawing.py')]:
        text=source(src);tree=ast.parse(text);parts=[]
        for node in tree.body:
            if isinstance(node,ast.Import)and any(a.name.startswith('pypdfium2')for a in node.names):continue
            if isinstance(node,ast.ImportFrom)and (node.module or '').startswith('rapid_doc'):continue
            if isinstance(node,ast.ImportFrom)and node.module=='enum_class':continue
            if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign)):
                snippet=ast.get_source_segment(text,node)
                snippet=snippet.replace('from rapid_doc.utils.pdf_image_text_probe import','from ._classification_probe import')
                if getattr(node,'name','')=='classify_pdf_by_page_chunks':continue
                parts.append(snippet)
        header='"""Original RapidDoc compatibility rules; native handles supplied by Rust.\nSource: '+src+'\nDo not replace with simplified heuristics.\n"""\n'
        if dst=='_drawing.py':
            enums=source('rapid_doc/utils/enum_class.py')
            for node in ast.parse(enums).body:
                if isinstance(node,ast.ClassDef)and node.name in ('BlockType','ContentType','SplitFlag'):header+=ast.get_source_segment(enums,node)+'\n'
            header+='class DataWriter:\n    pass\n'
        output=header+'\n\n'.join(parts)+'\n'
        (DEST/dst).write_text(output,encoding='utf-8')
        records[src]={'sha256':hashlib.sha256(text.encode()).hexdigest(),'target':dst,'target_sha256':hashlib.sha256(output.encode()).hexdigest()}
    for value in records.values():value['target_sha256']=hashlib.sha256((DEST/value['target']).read_bytes()).hexdigest()
    (ROOT/'packaging/compatibility-sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Copied original debug drawing adapters; classification/layout algorithms stay in Rust')
if __name__=='__main__':main()
