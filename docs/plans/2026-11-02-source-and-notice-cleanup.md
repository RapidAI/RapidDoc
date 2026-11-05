# Office output and migration notice cleanup

Goal: rename the Office output module and consolidate migration notices without changing parser behavior or losing attribution.

1. Rename `rust/rapiddoc-office/src/output` to `document_output`; update source, tests, generators and ignore rules. Keep a public `output` re-export for existing Rust callers.
2. Consolidate migration notices verbatim into `licenses/MIGRATION-NOTICES.txt`, with a JSON index recording original paths, byte ranges and SHA256. Preserve differing versions and every original third-party license file.
3. Validate and synchronize the bundle/index into the Python distribution; update install checks and compatibility audit references.
4. Run affected Office output tests, workspace binding checks, notice integrity tests and inspect a built wheel. No parsing algorithm or model changes; no CUDA/full corpus run is needed.

## Implementation and validation

- Office source directory is now `document_output`; internal callers, binding imports, examples, tests and Unicode generator point to it. The old public Rust `output` name is a compatibility re-export. Removed the obsolete Git ignore exception; the renamed source/resources are visible to Git.
- Archived 120 notice files into 62 verbatim sections (61 names; the two different PDFTEXT notices are both retained). All source bytes were compared against their indexed archive slices before the originals were removed. Original third-party license files remain separate.
- The canonical bundle/index live in root `licenses/`. The synchronization script validates hashes/ranges and generates copies for the wheel and four Rust crates, reducing 120 individual notice files to 12 bundle/index files. Keeping crate-local generated copies allows independent crate redistribution. Installation checks now refer to the bundle; a dedicated test checks all indexed sections.
- Passed six Rust Office output/HTML suites covering DOCX, DOC, PPT/PPTX and XLS/XLSX; two notice integrity tests; 119-module compatibility audit generation/check. All changed Python install scripts parse successfully.
- Built a Windows release wheel and installed it into the project `.venv`. Installed attribution checks passed. Existing Office installation validation passed on 23 DOCX reference documents, plus IO variants, batch, writers, data URI, empty input, error handling and PDF text smoke. No model load or original Python Office parser was used by that validation.
- No parsing algorithms were changed in this cleanup. Earlier CUDA sampling/long-document differences are separate unresolved work. Historical migration documents retain original paths; the notice index records those paths for lookup.
