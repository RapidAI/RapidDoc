# Python / Rust Python PDFium concurrency reproduction (before any fix)

No parser locks, business source files, installed packages or MQ services were
modified. Only an isolated test harness was added. Unsafe cases run in child
processes so a native crash does not take down an active service.

Environment: the business application's existing Python 3.12.10 environment,
installed RapidDoc 0.9.11 and pypdfium2 5.10.1. Rust's installed abi3 package was
loaded from the RapidDoc development environment by exact package path, keeping
all other dependencies in the same business interpreter. Rust uses its own
packaged PDFium and the prepared CPU ORT runtime. No GPU tests were performed.

Inputs: identical immutable bytes from `pdf_native.pdf`, `pdf_mixed.pdf` and
`pdf_scan.pdf` under `tests/migration/inputs`. Input sizes and SHA-256 hashes are
recorded in every case. Each route is compared with its own serial output;
Python and Rust output are not assumed to be identical.

## Results

| Case | Workers | Outcome |
| --- | --- | --- |
| Python host PDFium route only | 5 | 500/500, no errors or output changes |
| Python DocVortex PDFium route only | 5 | 500/500, no errors or output changes |
| Python host + DocVortex routes | 5 | Crashed with heap corruption `0xC0000374`; 177 recorded completions, 25 output changes before crash |
| Rust Python native text/render | 5 | 1000/1000, no errors or output changes |
| Python routes + Rust in one process | 5 | 473 Python-route exceptions, 38 output changes, process heap corruption; no Rust-route output mismatch recorded |
| Python high-level RapidDoc, serial | 1 | 12/12, no errors or output changes |
| Python high-level RapidDoc, parallel | 5 | First run completed 30 calls with 9 changed outputs, then heap corruption on exit; repeat crashed before first completion was recorded |
| Rust high-level RapidDoc, serial | 1 | 12/12, no errors or output changes |
| Rust high-level RapidDoc, per-thread instances | 5 | 60/60, no errors or output changes |
| Python + Rust high-level RapidDoc | 5 | Python output changed and access violation occurred; process subsequently crashed |
| Rust high-level RapidDoc, shared single instance | 5 | 1 successful call, 14 explicit `Document engine is already in use` rejections; process exited normally |

High-level cases use common rule layout, OCR and tables enabled, formula and
orientation disabled, and only the first page. They test PDFium concurrency and
OCR integration, not the application's complete Plus-L/Torch configuration or
an arbitrary real failing production document.

## Findings and boundaries

The two original Python routes independently serialize their calls but do not
serialize against each other. Their lock objects are different and a two-thread
probe confirmed both locks can be held simultaneously. The single-route control
tests pass while combining the routes reproducibly corrupts PDFium/heap state.
Python's high-level API also exposes the failure, so this is not merely a lock
identity test. A captured high-level mismatch returned empty Markdown and an
empty content list where its serial result contained text and a table.

Exceptions include `Failed to load page`, access violations and `Failed to load
document (PDFium: Success)`. The exact production message `Data format error`
was not captured in this run. Without the original failed PDF/traceback, the
production incident cannot be conclusively attributed to this race alone.

Rust's tested independent instances do not reproduce the failure. The shared
single-engine busy rejection is deliberate and must not be described as support
for simultaneous calls on one engine. A process containing unsafe original
Python calls can still crash the whole process, including any Rust callers.

An unrelated installed-preview API issue was discovered before the stress run:
explicit `table_config={"use_word_box": False}` fails serially in Rust with
`Unsupported model option: use_word_box`. Those two setup-failed runs were
excluded. Subsequent comparison cases omit the redundant key; both routes use
their existing false default. No production implementation was altered to make
the test pass.

## Reproduction and evidence

Harness: `scripts/migration/reproduce_pdfium_concurrency.py`.

```powershell
.venv/Scripts/python.exe scripts/migration/reproduce_pdfium_concurrency.py --case python --workers 5 --tasks 1000 --result build/pdfium-concurrency/repeat-python.json
.venv/Scripts/python.exe scripts/migration/reproduce_pdfium_concurrency.py --case rust --workers 5 --tasks 1000 --result build/pdfium-concurrency/repeat-rust.json
.venv/Scripts/python.exe scripts/migration/reproduce_pdfium_concurrency.py --case pipeline-rust --workers 5 --tasks 60 --result build/pdfium-concurrency/repeat-rust-api.json
```

`--python-route host|docvortex` isolates the two Python routes. `--shared-engine`
tests sharing one high-level instance. `--implementation workspace` switches
from the business installed Python implementation to the development source for
a later before/after check. Locks have not been unified yet.

Evidence: `build/pdfium-concurrency/summary.json`, per-case `.json` results,
`.process.json` process receipts and `.log` files containing fault-handler
native crash stacks. V2 runs additionally preserve serial output and the first
captured output mismatches.
