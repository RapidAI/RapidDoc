"""Download the pinned official KaTeX runtime for static review-page generation."""
import io
import tarfile
import urllib.request
from pathlib import Path


def main():
    target = Path(__file__).resolve().parents[1]/'build/eval-review-katex-0.16.22'
    target.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://registry.npmjs.org/katex/-/katex-0.16.22.tgz',timeout=60) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        # Read only named members; never extract archive paths to the filesystem.
        for name in ('package/dist/katex.js','package/LICENSE'):
            (target/Path(name).name).write_bytes(archive.extractfile(name).read())
    print(target)


if __name__=='__main__':main()
