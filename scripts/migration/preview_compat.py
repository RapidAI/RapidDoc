"""Explicit opt-in to source-tree migration checkpoints; never shipped in wheels."""
from pathlib import Path
import importlib
import sys


def activate():
    directory=str(Path(__file__).resolve().parent)
    if directory not in sys.path:sys.path.insert(0,directory)
    sdk=importlib.import_module('rapid_doc_rust')
    if getattr(sdk,'_migration_tools_active',False):return
    if not getattr(sdk._native,'MIGRATION_TOOLS',False):
        raise RuntimeError('Migration checkpoints require a source build with --features migration-tools; normal RapidDoc parsing does not need this feature')
    preview=importlib.import_module('rapid_doc_rust_preview')
    for name in dir(preview):
        if not name.startswith('__') and name not in ('json','math','sys','os','Path','dataclass'):
            setattr(sdk,name,getattr(preview,name))
    sdk._migration_tools_active=True
