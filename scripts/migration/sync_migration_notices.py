"""Validate the verbatim migration notice archive and sync it into the wheel."""
import argparse
import hashlib
import json
from pathlib import Path
from license_bundle import read_notice, read_tree, compact

ROOT = Path(__file__).resolve().parents[2]
FILENAMES = ('MIGRATION-NOTICES.txt', 'migration-notices.json')


def validate(bundle, index):
    if index['schema'] != 1:
        raise ValueError('Unsupported migration notice index')
    previous = 0
    paths = set()
    for entry in index['entries']:
        start, end = entry['start'], entry['end']
        if not 0 <= previous <= start <= end <= len(bundle):
            raise ValueError('Invalid notice byte range')
        raw = bundle[start:end]
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'Notice content changed: {entry["name"]}')
        for path in entry['original_paths']:
            if path in paths:
                raise ValueError(f'Duplicate original path: {path}')
            paths.add(path)
        previous = end
    if not paths:
        raise ValueError('Empty migration notice index')
    return len(paths)


def sync(check=False):
    source = ROOT / 'licenses'
    bundle = (source / FILENAMES[0]).read_bytes()
    index = json.loads((source / FILENAMES[1]).read_text(encoding='utf8'))
    count = validate(bundle, index)
    # Keep attribution in both wheels and independently packaged Rust crates.
    destinations = {ROOT / 'bindings/python/python/rapid_doc_rust/licenses'}
    destinations.update(ROOT / Path(path).parent for entry in index['entries']
                        for path in entry['original_paths'] if path.startswith('rust/'))
    for destination in sorted(destinations):
        sdk=destination==ROOT/'bindings/python/python/rapid_doc_rust/licenses'
        if sdk and check:
            from release_notice_policy import CARGO_NOTICES,load_policy
            expected=read_tree(CARGO_NOTICES)
            for path in load_policy()['retained_migration']:
                if read_notice(destination/path)!=expected[path]:
                    raise ValueError(f'Stale released attribution: {path}')
            continue
        for name in FILENAMES:
            if name=='migration-notices.json' and destination==ROOT/'bindings/python/python/rapid_doc_rust/licenses':
                continue  # Development byte-offset index is retained in the source tree.
            raw = (source / name).read_bytes()
            target = destination / name
            if check:
                if read_notice(target) != raw:
                    raise ValueError(f'Packaged migration notices are stale: {target}')
            else:
                target.write_bytes(raw)
        if not check and sdk:
            from release_notice_policy import filter_entries
            compact(destination,entries=filter_entries(read_tree(destination),'main','all')[0],shared_terms=True)
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(f'Validated {sync(args.check)} original notice paths')
