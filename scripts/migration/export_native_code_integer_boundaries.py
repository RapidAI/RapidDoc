"""Freeze Python integer arithmetic boundaries without changing earlier corpora."""
import copy
from common import ROOT, write_json
from export_native_code_containers import base, reference

def main():
    target = ROOT/'build/migration/native-code-integer-boundaries.json'
    if target.exists():
        raise FileExistsError(target)
    cases = []
    for mode in ['dark-min', 'light-max', 'large-source', 'negative-source']:
        q = base('filled')
        if mode in ['dark-min', 'light-max']:
            q['source']['path_infos'][0]['fill_rgba'] = [-(1 << 63) if mode == 'dark-min' else (1 << 63)-1, 255, 255, 255]
        else:
            for i, line in enumerate(q['source']['lines']):
                line['source_index'] = i*(1 << 58) if mode == 'large-source' else -(1 << 63)+i
                for ch in line['chars']:
                    ch.pop('char_idx', None)
        cases.append(dict(label=mode, request=copy.deepcopy(q), expected=reference(q)))
    write_json(target, dict(cases=cases, errors=[]))
    print(f'{len(cases)} old complete integer-boundary results')

if __name__ == '__main__':
    main()
