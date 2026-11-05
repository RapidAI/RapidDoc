"""Additional old glyph identity, blank-row, Unicode width and true multi-lane boundaries."""
import copy,json
from common import ROOT,write_json
from export_native_code_containers import base,glyphs,reference
def main():
    target=ROOT/'build/migration/native-code-container-boundaries.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):cases.append(dict(label=label,request=copy.deepcopy(q),expected=reference(q)))
    modes=['control-fallback','char-idx-missing','char-idx-string','char-idx-float','reverse-char-box','char-box-zero','wide','wide-ratio-low','wide-ratio-high','alphabetic-tight-gap','deep-blank-rows','nested-background','true-two-lanes','heading-cross-lanes','algorithm-border']
    for mode in modes:
        q=base('algorithm'if mode in ['true-two-lanes','heading-cross-lanes','algorithm-border']else'filled');s=q['source']
        if mode=='control-fallback':
            for l in s['lines']:l['text']='ab\x02\n\xad\t code\x00\u200b\u0085tail';l['chars']=[]
        if mode.startswith('char-idx-'):
            for l in s['lines']:
                for ch in l['chars']:
                    if mode=='char-idx-missing':ch.pop('char_idx',None)
                    if mode=='char-idx-string':ch['char_idx']=str(ch['char_idx'])
                    if mode=='char-idx-float':ch['char_idx']=float(ch['char_idx'])+.9
        if mode in ['reverse-char-box','char-box-zero']:
            for l in s['lines']:
                for ch in l['chars']:ch['bbox'][2]=ch['bbox'][0]-5. if mode=='reverse-char-box'else ch['bbox'][0]
        if mode.startswith('wide'):
            for l in s['lines']:l['text']='aa中中bb';l['bbox'][2]=l['bbox'][0]+40.
            glyphs(s)
            for l in s['lines']:
                for ch in l['chars']:
                    if ch['char']=='中':ch['bbox'][2]=ch['bbox'][0]+(6. if mode=='wide-ratio-low'else 12. if mode=='wide-ratio-high'else 10.)
        if mode=='alphabetic-tight-gap':
            for l in s['lines']:
                for ch in l['chars']:ch['tight_bbox']=[ch['bbox'][0]+.1,ch['bbox'][1],ch['bbox'][2]-.1,ch['bbox'][3]]
        if mode=='deep-blank-rows':
            for l in s['lines'][3:]:l['bbox'][1]+=30.;l['bbox'][3]+=30.
            glyphs(s)
        if mode=='nested-background':p=copy.deepcopy(s['path_infos'][0]);p['bbox']=[32.,92.,368.,248.];s['path_infos'].append(p)
        if mode in ['true-two-lanes','heading-cross-lanes']:
            s['page_size']=[800.,600.];left=copy.deepcopy(s['lines']);right=copy.deepcopy(left)
            for l in right:l['source_index']+=1000;l['bbox'][0]+=400.;l['bbox'][2]+=400.
            s['lines']=left+right
            if mode=='heading-cross-lanes':s['lines'][0]['bbox']=[40.,75.,700.,85.]
            glyphs(s)
        if mode=='algorithm-border':s['drawing_lines']=[dict(bbox=[40.,y,300.,y],orientation='horizontal',width=0.)for y in [100.,230.,245.]]
        add(mode,q)
    for text in ['f(x);','f();','x f ( (x));','function。x();','FUNCTION x','endé','end_','if\u0301','for中','while x','not syntax']:
        q=base('ruled')
        for l in q['source']['lines']:l['text']=text;l['font_signature']=['Body',0];l['chars']=[];l['median_glyph_width']=None
        add('syntax-'+repr(text),q)
    refs=json.loads((ROOT/'build/migration/native-code-containers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,scope='Old complete builders with actual two-lane/cross-lane layout, glyph source identity, blank rows, EAW ratio and fallback sanitization.',cases=cases,errors=[]));print('Frozen code boundaries',len(cases))
if __name__=='__main__':main()
