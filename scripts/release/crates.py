"""Package every Rust dependency together; upload only on explicit request."""

import argparse
import json
import os
import tomllib
import urllib.request
import urllib.error
from common import ROOT, OUT, check_version, run, sha, version

# Dependencies must be available in the registry before their consumers upload.
ORDER = [
    "rapiddoc-core",
    "rapiddoc-docvortex-core",
    "rapiddoc-docvortex-pdfium-paths",
    "rapiddoc-lopdf",
    "rapiddoc-native",
    "rapiddoc-pdf",
    "rapiddoc-inference",
    "rapiddoc-office",
    "rapiddoc-render",
    "rapiddoc",
    "rapiddoc-cli",
]
FORKS = {
    "rapiddoc-docvortex-core": "docvortex-core",
    "rapiddoc-docvortex-pdfium-paths": "docvortex-pdfium-paths",
    "rapiddoc-lopdf": "lopdf-f64",
}


def package_root(name):
    return ROOT / "rust" / ("vendor/" + FORKS[name] if name in FORKS else name)


def package_version(name):
    value = tomllib.loads((package_root(name) / "Cargo.toml").read_text("utf8"))["package"]["version"]
    return version() if isinstance(value, dict) else value


def published(name):
    # All release names are longer than three characters and use this sparse path.
    url = f"https://index.crates.io/{name[:2]}/{name[2:4]}/{name}"
    expected = package_version(name)
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            return any(json.loads(line)["vers"] == expected for line in response if line.strip())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def readiness():
    issues = []
    manifests = {
        name: tomllib.loads((package_root(name) / "Cargo.toml").read_text("utf8"))
        for name in ORDER
    }
    for name, manifest in manifests.items():
        if manifest["package"]["name"] != name:
            issues.append(f"{name}: package name differs from release plan")
        for section in ("dependencies", "build-dependencies"):
            for dependency, config in manifest.get(section, {}).items():
                if not isinstance(config, dict) or "path" not in config:
                    continue
                target = config.get("package", dependency)
                if "version" not in config:
                    issues.append(f"{name}: {dependency} needs a registry version")
                elif target not in manifests:
                    issues.append(f"{name}: {target} is missing from the release plan")
                else:
                    target_version = manifests[target]["package"]["version"]
                    if isinstance(target_version, dict):
                        target_version = version()
                    if config["version"] != "=" + target_version:
                        issues.append(
                            f"{name}: {dependency} registry version differs from source"
                        )
                    if (package_root(name) / config["path"]).resolve() != package_root(
                        target
                    ).resolve():
                        issues.append(
                            f"{name}: {dependency} path differs from release plan"
                        )
                    if ORDER.index(target) >= ORDER.index(name):
                        issues.append(f"{name}: {target} must publish first")
    # Do not scan Rust source with a regex: cfg(test) fixtures are not production
    # includes. Cargo's extracted-package build is the authoritative resource check.
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Local packaging only"
    )
    args = parser.parse_args()
    check_version(version(), args.publish)
    if args.publish and args.allow_dirty:
        raise SystemExit("Publishing requires a clean checkout")
    issues = readiness()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "ready": not issues,
        "issues": issues,
        "packages": ORDER,
        "verified": False,
    }
    report_path = OUT / "crates-readiness.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", "utf8")
    if issues:
        raise SystemExit(
            "crates.io release blocked before uploading:\n- " + "\n- ".join(issues)
        )
    if args.publish and not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise SystemExit("Configure CARGO_REGISTRY_TOKEN")
    # Package as a set: Cargo stages the unpublished local dependencies in a
    # temporary registry and verifies the extracted sources against that registry.
    # Individual cargo publish --dry-run cannot resolve first-release dependencies.
    run(
        "cargo",
        "package",
        "--locked",
        "--no-default-features",
        *(["--allow-dirty"] if args.allow_dirty else []),
        *[arg for name in ORDER for arg in ("-p", name)],
    )
    report["verified"] = True
    report["archives"] = {}
    for name in ORDER:
        manifest = tomllib.loads((package_root(name) / "Cargo.toml").read_text("utf8"))
        v = manifest["package"]["version"]
        if isinstance(v, dict):
            v = version()
        path = ROOT / "target/package" / f"{name}-{v}.crate"
        report["archives"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n", "utf8")
    if args.publish:
        for name in ORDER:
            if published(name):
                print("Already published:", name, package_version(name))
                continue
            run("cargo", "publish", "--locked", "--no-default-features", "-p", name)


if __name__ == "__main__":
    main()
