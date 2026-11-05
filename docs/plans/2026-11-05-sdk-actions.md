# SDK Actions implementation plan

**Goal:** Build and validate native SDK artifacts without credentials; publish
only through explicit, version-checked manual workflows.

**Architecture:** Reuse migration packagers on Windows/Linux. Separate native
platform packages from the npm loader. Keep the original Python release workflow
unchanged. Registry jobs receive credentials only after build jobs complete.

**Tech Stack:** GitHub Actions, Python release helpers, Cargo, maturin, Maven,
npm, Go, Central Portal API.

1. Add shared version/runtime/build helpers and seven-format installation checks.
2. Add npm platform package resolution and distribution assembly.
3. Add manual build/publication workflows, Central staging, Go/Release tags and
   a separate crates.io readiness/publishing workflow.
4. Validate workflow syntax, version guards, packaging and failure cases locally;
   document credentials and untested hosted-runner/platform prerequisites.

CUDA packaging uses explicitly configured self-hosted CUDA runners and a verified
operator bundle; no existing production server is registered as a runner.
Crates.io publication must fail before uploading anything while local forks,
path-only dependencies or out-of-package compile resources remain unresolved.
