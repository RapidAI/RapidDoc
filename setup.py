"""Tag wheels with native ops and exclude binaries for other platforms."""
import sys
import platform
from pathlib import Path
from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).parent
LIB = ROOT / "rapid_doc/backend/onnx_ops/lib"
SUFFIX = ".dll" if sys.platform == "win32" else ".so" if sys.platform == "linux" else None
if platform.machine().lower() not in ("amd64", "x86_64"):
    SUFFIX = None  # Bundled native artifacts currently target x86_64 only.


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return bool(SUFFIX and list(LIB.glob("*" + SUFFIX)))


class PlatformBuild(build_py):
    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        return [p for p in files if Path(p).suffix not in (".so", ".dll")
                or Path(p).suffix == SUFFIX]


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        if self.distribution.has_ext_modules():
            # The native library uses the ONNX Runtime C API, not the CPython ABI.
            return "py3", "none", plat
        return python, abi, plat


setup(distclass=NativeDistribution,
      cmdclass={"build_py": PlatformBuild, "bdist_wheel": PlatformWheel})
