"""Historical migration regressions opt into source-only diagnostic checkpoints."""
from pathlib import Path
import runpy
from rapid_doc_rust import _native
if getattr(_native,'MIGRATION_TOOLS',False):
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/migration/preview_compat.py'))['activate']()
