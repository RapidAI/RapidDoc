# Shared Python PDFium lock fix

Changed only the current RapidDoc Python source. The vendored DocVortex PDFium
runtime now aliases `rapid_doc.utils.PyPDFium2Parser.lock`; it no longer creates
a separate `threading.RLock`. Its existing guards continue covering font
initialization, native access and cleanup. The host lock is already reentrant,
so nested initialization and cross-wrapper calls are safe. No document-wide
lock was added around model inference, downloads or detached image processing.

The original business `.venv`, business source, MQ configuration and Rust
binding have not been modified or reinstalled. The business `.venv` vendor lock
file remains at SHA-256
`5d4aa4315406061505a611bfb451ce0b14c2d1fef5cd5a538ec2963c1803de42`.

## Before / after verification in the same workspace interpreter

The earlier reproduction used the business installed package. Before applying
this fix, the current workspace implementation was separately tested in its
existing interpreter: five threads combining the host and vendor PDFium routes
again produced changed output and heap corruption `0xC0000374`. Thus the
workspace before/after comparison does not depend on switching Python versions.

Three regression tests were added before the fix. The lock identity and
cross-thread exclusion tests failed on the old source; nested initialization /
exception cleanup passed. After the fix all three pass. No test replaces or
monkey-patches the lock implementation to obtain the result.

Five-thread stress after the fix:

| Case | Completed | Exceptions | Output changes | Process exit |
| --- | --- | --- | --- | --- |
| Host + DocVortex native reads/renders | 1000/1000 | 0 | 0 | normal |
| Python native routes + Rust Python, same process | 1000/1000 | 0 | 0 | normal |
| Python high-level RapidDoc | 60/60 | 0 | 0 | normal |
| Python + Rust high-level RapidDoc, independent instances | 60/60 | 0 | 0 | normal |

The native serial baseline hashes before and after the fix are identical.
Cases use the same immutable PDF inputs and each backend's own serial baseline;
this does not assume Python and Rust produce identical output. The high-level
scope remains common rule layout, OCR and tables, formula/orientation disabled,
first-page parsing. No full regression, Plus-L/Torch production-profile test or
CUDA run was performed. A future explicit failing production PDF still requires
its own test; the exact original `Data format error` message was not reproduced.

Evidence under `build/pdfium-concurrency`: `workspace-before-lock.*`,
`workspace-after-lock.*`, `workspace-mixed-after-lock.*`,
`workspace-pipeline-after-lock.*`, `workspace-pipeline-mixed-after-lock.*`,
`shared-lock-before.log`, `shared-lock-after.log`, and `lock-fix-summary.json`.

Re-run only the affected checks:

```powershell
.venv/Scripts/python.exe -m pytest tests/native_pdf/test_shared_pdfium_lock.py -q
.venv/Scripts/python.exe scripts/migration/reproduce_pdfium_concurrency.py --implementation workspace --case python --workers 5 --tasks 1000 --result build/pdfium-concurrency/repeat-after-fix.json
```

Rust single-instance concurrent-call rejection is unchanged. Independent Rust
instances are used in the coexistence tests; this fix does not alter that API's
concurrency contract.
