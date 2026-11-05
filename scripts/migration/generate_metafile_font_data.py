"""Extract Pillow's unchanged public-domain Aileron default subset for Rust."""
from pathlib import Path
import hashlib
from PIL import ImageFont,__version__
def main():
    if __version__!='12.2.0':raise RuntimeError('Expected reference Pillow 12.2.0')
    data=ImageFont.load_default(size=12).font_bytes
    path=Path(__file__).resolve().parents[2]/'rust/rapiddoc-office/src/metafile/aileron-default.ttf'
    path.write_bytes(data)
    print('Pillow Aileron subset:',len(data),'bytes; SHA256',hashlib.sha256(data).hexdigest())
if __name__=='__main__':main()
