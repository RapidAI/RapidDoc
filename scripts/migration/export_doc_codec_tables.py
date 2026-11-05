"""Freeze Windows DOC DBCS semantics separately from the XLS codec aliases."""
import codecs
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    directory=ROOT/'rust/rapiddoc-office/src/doc/codepages';directory.mkdir(exist_ok=True)
    for cp in [932,949,950]:
        decoder=codecs.getincrementaldecoder('cp'+str(cp))(errors='replace')
        states=[decoder.getstate()];lookup={states[0]:0};outputs=[''];intern={'':0};transitions=[];final=[]
        def symbol(text):
            if text not in intern:intern[text]=len(outputs);outputs.append(text)
            return intern[text]
        index=0
        while index<len(states):
            state=states[index];decoder.setstate(state);final.append(symbol(decoder.decode(b'',final=True)))
            for value in range(256):
                decoder.setstate(state);text=decoder.decode(bytes([value]),final=False);new=decoder.getstate()
                if new not in lookup:lookup[new]=len(states);states.append(new)
                transitions.append((lookup[new],symbol(text)))
            index+=1
        raw=json.dumps(dict(symbols=outputs,transitions=transitions,final=final),ensure_ascii=True,separators=(',',':')).encode()
        (directory/f'{cp}.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        print('DOC CP',cp,'states',len(states))


if __name__=='__main__':main()
