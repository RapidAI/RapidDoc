"""Share identical license words, retaining each attribution and exact layout.

No SPDX-name-only matching: an entire template must match word-for-word,
including punctuation and case. Unknown or modified clauses remain verbatim.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def templates():
    mit=(ROOT/'licenses/NUM-RATIONAL-LICENSE-MIT.txt').read_bytes()
    mit=mit[mit.index(b'Permission is hereby granted'):].strip()
    apache=(ROOT/'licenses/NUM-RATIONAL-LICENSE-APACHE.txt').read_bytes()
    start=apache.index(b'TERMS AND CONDITIONS FOR USE')
    end=apache.index(b'END OF TERMS AND CONDITIONS')+len(b'END OF TERMS AND CONDITIONS')
    python=(ROOT/'licenses/PYTHON-CODECS-LICENSE.txt').read_bytes().strip()
    return {'MIT':mit,'Apache-2.0':apache[start:end],'CPython-license-chain':python}


def pack_shared(entries, legal_templates=None):
    legal_templates=templates() if legal_templates is None else legal_templates
    matchers={name:re.compile(rb'\s+'.join(re.escape(token) for token in re.findall(rb'\S+',raw)))
              for name,raw in legal_templates.items()}
    groups={}
    for path,data in sorted(entries.items()):
        data.decode('utf8')
        group=groups.setdefault(sha(data),{'data':data,'paths':[]})
        group['paths'].append(path)
    records=[];used={};layouts={}
    for checksum,group in groups.items():
        data=group['data'];parts=[];cursor=0
        matches=sorted((match.start(),match.end(),name) for name,regex in matchers.items()
                       for match in regex.finditer(data))
        for start,end,name in matches:
            if start<cursor:continue
            if start>cursor:parts.append({'literal':data[cursor:start]})
            body=data[start:end]
            separators=re.findall(rb'\s+',body)
            layout=[[i,s.decode('ascii')] for i,s in enumerate(separators) if s!=b' ']
            key=sha(json.dumps(layout,separators=(',',':')).encode())
            layouts[key]=layout;used[name]=legal_templates[name]
            parts.append({'term':name,'layout':key})
            cursor=end
        if cursor<len(data):parts.append({'literal':data[cursor:]})
        records.append({'sha256':checksum,'original_paths':group['paths'],'parts':parts})
    out=bytearray(b'RapidDoc third-party licenses and copyright notices\n'
                  b'Full shared terms appear once below. Each component retains its own attribution.\n'
                  b'Only identical words are shared; modified/unknown license text remains complete.\n')
    terms={}
    for name,data in sorted(used.items()):
        out.extend(('\n=== Full license terms: '+name+' ===\n\n').encode())
        start=len(out);out.extend(data)
        terms[name]={'start':start,'end':len(out),'sha256':sha(data)}
        out.extend(b'\n')
    for record in records:
        out.extend(('\n=== Components: '+', '.join(record['original_paths'])+' ===\n').encode())
        for part in record['parts']:
            if 'literal' in part:
                raw=part.pop('literal');start=len(out);out.extend(raw)
                part.update(start=start,end=len(out))
            else:
                out.extend(('\n[Full '+part['term']+' terms above apply here.]\n').encode())
        out.extend(b'\n')
    index=json.dumps({'schema':'rapiddoc.notice-bundle.v2','terms':terms,'layouts':layouts,'entries':records},
                     ensure_ascii=False,separators=(',',':')).encode()+b'\n'
    if unpack_shared(bytes(out),index)!=entries:
        raise ValueError('Shared license text did not preserve original clauses/attributions')
    return bytes(out),index


def unpack_shared(text,index):
    raw = bytes(text)
    try:
        return _unpack_shared(raw, index)
    except ValueError:
        # Git may check generated bundles out with CRLF on Windows, while
        # their stored offsets and hashes use canonical LF bytes.
        if b'\r\n' not in raw:
            raise
        return _unpack_shared(raw.replace(b'\r\n', b'\n'), index)


def _unpack_shared(text,index):
    metadata=json.loads(index);terms={}
    def slice_checked(record):
        start,end=record['start'],record['end']
        if not 0<=start<=end<=len(text):raise ValueError('Invalid notice offsets')
        return text[start:end]
    for name,record in metadata['terms'].items():
        raw=slice_checked(record)
        if sha(raw)!=record['sha256']:raise ValueError('Shared license terms changed')
        terms[name]=re.findall(rb'\S+',raw)
    result={}
    for record in metadata['entries']:
        data=bytearray()
        for part in record['parts']:
            if 'term' not in part:
                data.extend(slice_checked(part));continue
            tokens=terms[part['term']]
            layout=dict(metadata['layouts'][part['layout']])
            for i,token in enumerate(tokens):
                if i:data.extend(layout.get(i-1,' ').encode('ascii'))
                data.extend(token)
        raw=bytes(data)
        if sha(raw)!=record['sha256']:raise ValueError('Component copyright/license text changed')
        for name in record['original_paths']:
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or name in result:raise ValueError('Invalid/duplicate notice path')
            result[name]=raw
    if not result:raise ValueError('Empty notice bundle')
    return result
