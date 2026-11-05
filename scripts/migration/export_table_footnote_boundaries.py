"""Additional immutable angle, glyph reference and chain limit facts."""
import copy
from common import ROOT,digest,write_json
from export_native_table_footnotes import base,reference
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-table-footnote-boundaries.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    cases=[]
    def add(label,q):q=clean(q);cases.append(dict(label=label,request=q,expected=reference(q)))
    for angle in [0,90,180,270,15]:
        for kind in ['explicit','compact','superscript']:
            q=base('Note: values'if kind=='explicit'else'a) values');q['angle']=angle
            if kind=='superscript':q['lines'][0]['text']='Long ordinary cell without compact tokens';q['lines'][0]['chars']=[dict(char='x',bbox=[50.,110.,60.,120.]),dict(char='a',bbox=[60.,108.,65.,114.])]
            for l in q['lines']:
                l['angle']=angle;l['bbox']=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),angle)
                for c in l['chars']:c['bbox']=_rotate_bbox_from_upright(tuple(c['bbox']),tuple(q['page_size']),angle)
            add(f'angle-{angle}-{kind}',q)
    for bottom in [299.99999,300.,300.00001]:
        q=base();f=q['rows'][2]['fragments'][0];f['local_bbox']=[40.,290.,200.,bottom];q['rows'][2]['bbox']=f['local_bbox'];q['rows'][2]['center_y']=(290.+bottom)/2
        # Selected intervening core row advances bottom without starting a new chain.
        middle=copy.deepcopy(q['rows'][0]);middle['bbox']=[40.,215.,200.,290.];middle['fragments'][0]['local_bbox']=middle['bbox'];q['rows'].insert(2,middle);add('chain-10em-'+str(bottom),q)
    q=base();q['rows'][1]['bbox'][3]=1300.;q['rows'][1]['fragments'][0]['local_bbox'][3]=1300.;add('first-row-no-chain-cap',q)
    for gap in [9.99999,10.,10.00001]:
        q=base();q['rows'][2]['bbox'][1]=215.+gap;q['rows'][2]['bbox'][3]=225.+gap;q['rows'][2]['fragments'][0]['local_bbox']=q['rows'][2]['bbox'];add('continuation-gap-'+str(gap),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'angle/chain note cases;',sum(bool(c['expected']['footnote_rows'])for c in cases),'positive')
if __name__=='__main__':main()
