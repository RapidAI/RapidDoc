"""Notice packaging must preserve attribution bytes and every parser binary."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/migration'))
from license_bundle import TEXT, INDEX, pack, unpack, read_tree, compact, read_notice
from compact_release_notices import repack


def test_dedup_preserves_distinct_copyrights_and_original_bytes():
    entries = {'one/LICENSE': b'Copyright One\r\nMIT\n',
               'two/LICENSE': b'Copyright One\r\nMIT\n',
               'three/LICENSE': b'Copyright Three\nMIT'}
    text, index = pack(entries)
    assert len(json.loads(index)['entries']) == 2
    assert text.count(b'Copyright One') == 1
    assert unpack(text, index) == entries
    assert b'one/LICENSE' in text and b'two/LICENSE' in text


@pytest.mark.parametrize('damage', ['content', 'offset', 'traversal'])
def test_corrupt_bundle_is_rejected(damage):
    text, index = pack({'one/LICENSE': b'MIT with attribution'})
    data = json.loads(index)
    if damage == 'content':
        text = text.replace(b'MIT with attribution', b'ABC with attribution')
    elif damage == 'offset':
        data['entries'][0]['end'] = len(text)+1
    else:
        data['entries'][0]['original_paths'] = ['../elsewhere']
    with pytest.raises(ValueError):
        unpack(text, json.dumps(data).encode())


def test_compaction_is_idempotent_and_can_resolve_former_paths():
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
        root = Path(directory).resolve()
        assert root.is_relative_to(ROOT)
        (root/'a').mkdir()
        (root/'a/LICENSE').write_bytes(b'license A')
        (root/'B.txt').write_bytes(b'license A')
        original = read_tree(root)
        assert compact(root)['after'] == 2
        assert read_tree(root) == original
        assert read_notice(root/'a/LICENSE') == b'license A'
        before = {p.name:p.read_bytes() for p in root.iterdir()}
        compact(root)
        assert before == {p.name:p.read_bytes() for p in root.iterdir()}
        (root/'a').mkdir()
        (root/'a/LICENSE').write_bytes(b'updated license A')
        compact(root)
        assert read_notice(root/'a/LICENSE') == b'updated license A'
        assert read_notice(root/'B.txt') == b'license A'


def test_nested_staging_bundles_override_previous_package_bundle():
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
        root=Path(directory).resolve()
        assert root.is_relative_to(ROOT)
        text,index=pack({'office/LICENSE':b'old'})
        (root/TEXT).write_bytes(text);(root/INDEX).write_bytes(index)
        (root/'office').mkdir()
        text,index=pack({'LICENSE':b'new'})
        (root/'office'/TEXT).write_bytes(text);(root/'office'/INDEX).write_bytes(index)
        compact(root)
        assert read_tree(root)=={'office/LICENSE':b'new'}


def test_non_utf8_notice_keeps_its_original_file_and_encoding():
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
        root=Path(directory).resolve()
        assert root.is_relative_to(ROOT)
        original={'MIT.txt':b'MIT attribution','freetype.txt':b'Copyright \xa9 original license'}
        for name,data in original.items():(root/name).write_bytes(data)
        assert compact(root)['after']==3
        assert (root/'freetype.txt').read_bytes()==original['freetype.txt']
        assert read_tree(root)==original


def test_wheel_repack_preserves_runtime_and_parser_and_updates_record(tmp_path,monkeypatch):
    import compact_release_notices
    monkeypatch.setattr(compact_release_notices,'filter_entries',lambda entries,*args,**kwargs:(entries,[]))
    source=tmp_path/'original.whl';target=tmp_path/'compact.whl'
    entries={'rapid_doc_rust/_native.pyd':b'opaque native binary\x00\xff',
             'rapid_doc_rust/runtime/rapiddoc_pdfium.dll':b'unchanged PDFium',
             'rapid_doc_rust/__init__.py':b'# unchanged Python SDK\n',
             'rapid_doc_rust/licenses/a.txt':b'license A',
             'rapid_doc_rust/licenses/b.txt':b'license A',
             'rapid_doc_rust/runtime/licenses/pdfium.txt':b'license PDFium',
             'rapid_doc_rust/runtime/manifest.json':b'{"library":"rapiddoc_pdfium.dll","license_files":{}}',
             'rapid_doc_rust-0.1.0.dist-info/RECORD':b''}
    with zipfile.ZipFile(source,'w') as wheel:
        for name,data in entries.items():wheel.writestr(name,data)
    result=repack(source,target)
    assert result['parser_and_runtime_bytes_unchanged']
    with zipfile.ZipFile(target) as wheel:
        for name in ['rapid_doc_rust/_native.pyd','rapid_doc_rust/runtime/rapiddoc_pdfium.dll','rapid_doc_rust/__init__.py']:
            assert wheel.read(name)==entries[name]
        prefix='rapid_doc_rust/licenses/'
        assert unpack(wheel.read(prefix+TEXT),wheel.read(prefix+INDEX))=={'a.txt':b'license A','b.txt':b'license A'}
        record=list(csv.reader(io.StringIO(wheel.read('rapid_doc_rust-0.1.0.dist-info/RECORD').decode())))
        assert len(record)==len(wheel.namelist())
        for name,checksum,length in record:
            if checksum:assert len(wheel.read(name))==int(length)
        manifest=json.loads(wheel.read('rapid_doc_rust/runtime/manifest.json'))
        assert set(manifest['license_files'])=={'licenses/'+TEXT,'licenses/'+INDEX}


def test_release_policy_removes_only_recorded_nonruntime_and_alternative_notices():
    from release_notice_policy import filter_entries
    policy={'profiles':{'windows':['image 1']},'files':{
        'office/image/LICENSE-MIT':{'component':'image 1','selected':True},
        'office/image/LICENSE-APACHE':{'component':'image 1','selected':False},
        'office/syn/LICENSE-MIT':{'component':'syn 1','selected':True}}}
    entries={'office/image/LICENSE-MIT':b'copyright image MIT',
             'office/image/LICENSE-APACHE':b'Apache alternative',
             'office/syn/LICENSE-MIT':b'compiler-only syn',
             'office/office-native-provenance.json':b'development hashes',
             'unknown-NOTICE.txt':b'copyright must be preserved'}
    selected,removed=filter_entries(entries,'main','windows',policy=policy,generated={},office={})
    assert selected=={'office/image/LICENSE-MIT':entries['office/image/LICENSE-MIT'],
                      'unknown-NOTICE.txt':entries['unknown-NOTICE.txt']}
    assert len(removed)==3 and all(item['sha256'] for item in removed)


@pytest.mark.parametrize('profile',['windows','linux'])
def test_pdfium_platform_filter_keeps_shared_non_musl_notices(profile):
    from release_notice_policy import filter_entries
    policy={'profiles':{profile:[]},'files':{}}
    entries={name:b'copyright '+name.encode() for name in [
        'BUILD_LICENSES/android/libunwind.txt','BUILD_LICENSES/pdfium-binaries_musl/musl.txt',
        'BUILD_LICENSES/cibw_musl/libgcc_and_libstdc++.txt',
        'BUILD_LICENSES/all_except_musl/llvm-libc.txt','BUILD_LICENSES/freetype.txt']}
    selected,removed=filter_entries(entries,'pdfium',profile,policy=policy)
    assert set(selected)=={'BUILD_LICENSES/all_except_musl/llvm-libc.txt','BUILD_LICENSES/freetype.txt'}
    assert len(removed)==3


@pytest.mark.parametrize('expression,keep_apache',[('MIT OR Apache-2.0',False),('MIT AND Apache-2.0',True)])
def test_license_selection_respects_and_vs_or(tmp_path,expression,keep_apache):
    from generate_release_notice_policy import selected_files
    (tmp_path/'Cargo.toml').write_text('')
    for name in ['LICENSE-MIT','LICENSE-APACHE','NOTICE','COPYRIGHT']:(tmp_path/name).write_text(name)
    files,selected=selected_files({'manifest_path':str(tmp_path/'Cargo.toml'),'license':expression})
    assert 'LICENSE-MIT' in selected
    assert ('LICENSE-APACHE' in selected)==keep_apache
    assert {'NOTICE','COPYRIGHT'}<=selected


def test_shared_terms_keep_individual_copyrights_and_exact_original_layouts():
    from shared_license_terms import pack_shared,unpack_shared
    terms=b'Permission is hereby granted.\nNO WARRANTY.'
    entries={'one/LICENSE':b'Copyright One\n\n'+terms,
             'two/LICENSE':b'Copyright Two\r\nPermission  is\r\nhereby granted.\r\nNO WARRANTY.\n'}
    text,index=pack_shared(entries,{'MIT':terms})
    assert text.count(b'Permission is hereby granted.')==1
    assert b'Copyright One' in text and b'Copyright Two' in text
    assert unpack_shared(text,index)==entries
    assert unpack(text,index)==entries


def test_modified_or_unknown_license_terms_are_not_replaced_by_a_standard_name():
    from shared_license_terms import pack_shared,unpack_shared
    terms=b'Permission is hereby granted.\nNO WARRANTY.'
    original=b'Copyright Author\nPermission is hereby DENIED. NO WARRANTY.'
    text,index=pack_shared({'unknown/LICENSE':original},{'MIT':terms})
    assert original in text
    assert not json.loads(index)['terms']
    assert unpack_shared(text,index)=={'unknown/LICENSE':original}


def test_shared_term_corruption_is_rejected():
    from shared_license_terms import pack_shared,unpack_shared
    text,index=pack_shared({'one/LICENSE':b'Copyright Author\nCommon terms apply.'},
                           {'test':b'Common terms apply.'})
    with pytest.raises(ValueError,match='terms changed'):
        unpack_shared(text.replace(b'Common terms apply.',b'Changed terms apply.'),index)
