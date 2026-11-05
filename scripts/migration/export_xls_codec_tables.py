"""Compile original Python legacy codecs into native byte-state tables.

Includes replacement/continuation behavior, rather than assuming WHATWG codecs
are interchangeable with CPython's CP874/Shift-JIS/GBK/EUC-KR/Big5 decoders.
"""
import codecs,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    target=ROOT/'rust/rapiddoc-office/src/xls/codepages';target.mkdir(exist_ok=True)
    for cp in [874,932,936,949,950,*range(1250,1259)]:
        name={874:'cp874',932:'shift_jis',936:'gbk',949:'euc_kr',950:'big5'}.get(cp,'cp'+str(cp))
        if cp==949:
            single=[];pairs=[];hangul={}
            for a in range(256):
                try:single.append(bytes([a]).decode(name,'strict'))
                except UnicodeDecodeError:single.append(None)
            for a in range(256):
                for b in range(256):
                    if single[a] is not None:pairs.append([1,single[a]]);continue
                    try:pairs.append([2,bytes([a,b]).decode(name,'strict')])
                    except UnicodeDecodeError as e:pairs.append([e.end,'\ufffd'])
            initials={};vowels={};finals={}
            for c in range(0xac00,0xd7a4):
                raw=chr(c).encode(name)
                if len(raw)==8:
                    offset=c-0xac00
                    initials[offset//588]=raw[2:4];vowels[(offset//28)%21]=raw[4:6];finals[offset%28]=raw[6:8]
            assert len(initials)==19 and len(vowels)==21 and len(finals)==28
            # The eight-byte spelling also decodes to syllables that have a
            # shorter canonical KS X 1001 encoding. Include all 11,172 forms.
            for c in range(0xac00,0xd7a4):
                offset=c-0xac00;raw=b'\xa4\xd4'+initials[offset//588]+vowels[(offset//28)%21]+finals[offset%28]
                assert raw.decode(name)==chr(c)
                hangul[raw.hex()]=chr(c)
            data=json.dumps(dict(single=single,pairs=pairs,hangul=hangul),ensure_ascii=True,separators=(',',':')).encode()
            dest=target/(str(cp)+'.json.gz');dest.write_bytes(gzip.compress(data,mtime=0));print(cp,'composed Hangul',len(hangul),'bytes',dest.stat().st_size,flush=True);continue
        decoder=codecs.getincrementaldecoder(name)(errors='replace')
        states=[decoder.getstate()];lookup={states[0]:0};outputs=[''];intern={'':0};transitions=[];final=[]
        def symbol(text):
            if text not in intern:intern[text]=len(outputs);outputs.append(text)
            return intern[text]
        index=0
        while index<len(states):
            state=states[index]
            decoder.setstate(state);final.append(symbol(decoder.decode(b'',final=True)))
            for value in range(256):
                decoder.setstate(state);text=decoder.decode(bytes([value]),final=False);new=decoder.getstate()
                if new not in lookup:lookup[new]=len(states);states.append(new)
                transitions.append((lookup[new],symbol(text)))
            index+=1
        data=json.dumps(dict(symbols=outputs,transitions=transitions,final=final),ensure_ascii=True,separators=(',',':')).encode()
        dest=target/(str(cp)+'.json.gz');dest.write_bytes(gzip.compress(data,mtime=0))
        print(cp,'states',len(states),'bytes',dest.stat().st_size,flush=True)
if __name__=='__main__':main()
