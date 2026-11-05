"""Check the native original sorting backend before building the full engine."""
from pathlib import Path
import ctypes,json,os,subprocess,sys,shutil
import numpy as np
from setuptools._distutils._msvccompiler import _get_vc_env
root=Path(__file__).resolve().parents[2];out=root/'build/numpy-sort-probe';out.mkdir(exist_ok=True)
env=dict(os.environ);env.update(_get_vc_env('x86_amd64'))
compiler=shutil.which('cl.exe',path=env.get('path',env.get('PATH')))
if not compiler:raise FileNotFoundError('MSVC compiler unavailable')
source=root/'rust/rapiddoc-inference/src/numpy_argsort.cpp'
for isa,flags in [('avx2',['/arch:AVX2','/D__AVX2__']),('avx512',['/arch:AVX512','/D__AVX512DQ__','/D__AVX512VL__','/D__AVX512BW__'])]:
    dll=out/f'{isa}.dll'
    cmd=[compiler,'/nologo','/std:c++17','/EHsc','/O2','/LD',*flags,f'/DRAPIDDOC_SORT_NAME=rd_sort_{isa}',f'/I{root}/rust/rapiddoc-inference/vendor/x86-simd-sort/src',str(source),f'/Fo{out}/{isa}.obj','/link',f'/OUT:{dll}']
    p=subprocess.run(cmd,env=env,cwd=out,capture_output=True,text=True,errors='replace')
    (out/f'{isa}-compile.txt').write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    if isa=='avx512'and not np._core._multiarray_umath.__cpu_features__.get('AVX512_SKX'):continue
    lib=ctypes.CDLL(str(dll));call=getattr(lib,f'rd_sort_{isa}');call.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_size_t),ctypes.c_size_t]
    rng=np.random.default_rng(824)
    for n in [0,1,2,7,8,14,16,17,32,63,64,127,128,255,256,257,511,1000]:
        for kind in range(20):
            values=rng.integers(1,17,n).astype('float64') if kind%2 else rng.uniform(.2,30,n)
            if kind==1:values.sort()
            indices=np.arange(n,dtype=np.uintp);expected=np.argsort(values)
            assert call(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),n)==0
            assert np.array_equal(indices,expected),(isa,n,kind,indices[:20],expected[:20])
    print('Original NumPy native sorting backend matched:',isa,flush=True)
