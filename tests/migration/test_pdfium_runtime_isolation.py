"""PDFium path staging without loading or initializing native libraries."""
import concurrent.futures
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pdfium_isolation_test_module',
    ROOT/'bindings/python/python/rapid_doc_rust/_pdfium_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class PdfiumIsolation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root/'pdfium.dll'
        self.source.write_bytes(b'isolated-pdfium-fixture')
        runtime._COPIES.clear()
        self.cache = patch.object(runtime, '_cache_root', return_value=self.root/'cache')
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.environment = patch.dict(os.environ, {'RAPIDDOC_PDFIUM_LIBRARY': ''})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_default_uses_packaged_private_library(self):
        self.assertEqual(runtime.resolve_pdfium_library(), str(runtime._PACKAGED))
        self.assertFalse((self.root/'cache').exists())

    def test_external_copy_is_distinct_and_shared_by_identical_contents(self):
        before = self.source.read_bytes()
        copied = Path(runtime.resolve_pdfium_library(self.source))
        self.assertFalse(copied.samefile(self.source))
        self.assertEqual(copied.read_bytes(), before)
        self.assertEqual(self.source.read_bytes(), before)
        second = self.root/'other.dll'
        second.write_bytes(before)
        self.assertEqual(runtime.resolve_pdfium_library(second), str(copied))

    def test_environment_override_is_also_isolated(self):
        with patch.dict(os.environ, {'RAPIDDOC_PDFIUM_LIBRARY': str(self.source)}):
            self.assertFalse(Path(runtime.resolve_pdfium_library()).samefile(self.source))

    def test_concurrent_first_use_publishes_one_complete_file(self):
        with patch.object(runtime.shutil, 'copyfile', wraps=runtime.shutil.copyfile) as copy:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(runtime.resolve_pdfium_library, [self.source]*12))
            self.assertEqual(len(set(paths)), 1)
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(Path(paths[0]).read_bytes(), self.source.read_bytes())

    def test_invalid_cached_file_and_hardlink_are_replaced(self):
        copied = Path(runtime.resolve_pdfium_library(self.source))
        copied.write_bytes(b'incomplete')
        runtime._COPIES.clear()
        self.assertEqual(Path(runtime.resolve_pdfium_library(self.source)).read_bytes(), self.source.read_bytes())
        runtime._COPIES.clear()
        copied.unlink()
        os.link(self.source, copied)
        self.assertTrue(copied.samefile(self.source))
        self.assertFalse(Path(runtime.resolve_pdfium_library(self.source)).samefile(self.source))

    def test_missing_source_does_not_publish_a_copy(self):
        with self.assertRaises(FileNotFoundError):
            runtime.resolve_pdfium_library(self.root/'missing.dll')
        self.assertFalse((self.root/'cache').exists())


if __name__ == '__main__':
    unittest.main()
