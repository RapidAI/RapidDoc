"""Generate/check the Python baseline casefold data without linking Python in Rust."""
import argparse
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if unicodedata.unidata_version != "16.0.0":
        raise RuntimeError("Use the frozen Python Unicode 16.0.0 baseline to regenerate Office data")
    mapping = {chr(i): chr(i).casefold() for i in range(sys.maxunicode + 1) if chr(i) != chr(i).casefold()}
    data = (json.dumps(mapping, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    path = ROOT / "rust/rapiddoc-office/src/unicode_casefold.json"
    if args.check:
        assert path.read_bytes() == data, "Office casefold data differs from this Python baseline"
    else:
        path.write_bytes(data)
    zeroes = [i for i in range(sys.maxunicode + 1) if unicodedata.decimal(chr(i), None) == 0]
    decimal_data = ("// Generated from Python Unicode 16.0.0; see licenses/PYTHON-UNICODE.txt.\n"
                    "#[rustfmt::skip]\npub(super) const DECIMAL_ZEROES: &[u32] = &[" + ",".join(hex(i) for i in zeroes) + "];\n").encode()
    ranges=[]
    for i in range(sys.maxunicode+1):
        if chr(i).isdigit() and not chr(i).isdecimal():
            if ranges and i==ranges[-1][1]+1:ranges[-1][1]=i
            else:ranges.append([i,i])
    decimal_data+=("\n// Python Unicode 16.0.0 str.isdigit minus decimal, for page-number filtering.\n"
                   "#[rustfmt::skip]\npub(super) const ISDIGIT_EXTRA:&[(u32,u32)]=&["+
                   ",".join(f"(0x{a:x},0x{b:x})" for a,b in ranges)+"];\n").encode()
    decimal_path = ROOT / "rust/rapiddoc-office/src/docx/decimal_data.rs"
    if args.check:
        assert decimal_path.read_bytes() == decimal_data, "Office decimal data differs from frozen Python"
    else:
        decimal_path.write_bytes(decimal_data)
    forbidden=''.join(chr(i) for i in range(sys.maxunicode+1)
                      if i>127 and any(c in unicodedata.normalize('NFKC',chr(i)) for c in '/?#@:'))
    link_data=(json.dumps(forbidden,ensure_ascii=True)+'\n').encode()
    link_path=ROOT/'rust/rapiddoc-office/src/link_nfkc_forbidden.json'
    if args.check:
        assert link_path.read_bytes()==link_data,'Office link NFKC data differs from frozen Python'
    else:
        link_path.write_bytes(link_data)
    print(f"Office casefold: Unicode {unicodedata.unidata_version}, {len(mapping)} nonidentity entries")


if __name__ == "__main__":
    main()
