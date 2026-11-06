"""Assemble, sign and upload one complete Maven Central Portal deployment."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import uuid
import zipfile
import xml.etree.ElementTree as ET

from common import NS, OUT, check_version, run, version, ort_package_version


def assemble(artifacts, destination):
    files = {}
    for p in artifacts.glob("*/java/*"):
        if p.suffix not in [".jar", ".pom"]:
            continue
        data = p.read_bytes()
        if p.suffix == ".pom":
            data = data.replace(b"\r\n", b"\n")
        if p.name in files and files[p.name] != data:
            # Main JVM JARs include build timestamps; publish one platform's JVM
            # build. Native classifier artifacts must never collide.
            if p.name.startswith(f"rapid-doc-{version()}") and p.suffix == ".jar":
                continue
            raise ValueError("Conflicting Maven artifact: " + p.name)
        files[p.name] = data
    required = [
        f"rapid-doc-{version()}.jar",
        f"rapid-doc-{version()}-sources.jar",
        f"rapid-doc-{version()}-javadoc.jar",
        f"rapid-doc-{version()}.pom",
        f"rapid-doc-native-{version()}.pom",
        f"rapid-doc-ort-{ort_package_version()}.pom",
    ]
    for platform in ["windows-x86_64", "linux-x86_64", "macos-x86_64", "macos-aarch64", "linux-aarch64", "windows-aarch64"]:
        required += [
            f"rapid-doc-native-{version()}-{platform}.jar",
            f"rapid-doc-ort-{ort_package_version()}-{platform}.jar",
        ]
    if any(name not in files for name in required):
        raise ValueError("Incomplete Maven release; required: " + ", ".join(required))
    for filename, data in files.items():
        if filename.endswith(".pom"):
            p = ET.fromstring(data)
            group = p.findtext("m:groupId", namespaces=NS)
            artifact = p.findtext("m:artifactId", namespaces=NS)
            release = p.findtext("m:version", namespaces=NS)
            for name, content in files.items():
                if name == filename or (
                    name.endswith(".jar") and name.startswith(f"{artifact}-{release}")
                ):
                    target = (
                        destination
                        / group.replace(".", "/")
                        / artifact
                        / release
                        / name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--publish", action="store_true")
    a = p.parse_args()
    check_version(version(), a.publish)
    staging = OUT / "central"
    staging.mkdir(parents=True, exist_ok=False)
    assemble(a.artifacts, staging)
    if not a.publish:
        print("Central staging validated:", staging)
        return
    names = [
        "MAVEN_CENTRAL_USERNAME",
        "MAVEN_CENTRAL_PASSWORD",
        "MAVEN_GPG_PRIVATE_KEY",
        "MAVEN_GPG_PASSPHRASE",
    ]
    if any(not os.environ.get(name) for name in names):
        raise ValueError(
            "Configure all four MAVEN_* secrets; credentials are never printed"
        )
    with tempfile.TemporaryDirectory() as home:
        env = {**os.environ, "GNUPGHOME": home}
        os.chmod(home, 0o700)
        run(
            "gpg",
            "--batch",
            "--import",
            input=os.environ["MAVEN_GPG_PRIVATE_KEY"].encode(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for f in list(staging.rglob("*")):
            if not f.is_file():
                continue
            run(
                "gpg",
                "--batch",
                "--yes",
                "--armor",
                "--pinentry-mode",
                "loopback",
                "--passphrase-fd",
                "0",
                "--detach-sign",
                f,
                input=os.environ["MAVEN_GPG_PASSPHRASE"].encode() + b"\n",
                env=env,
            )
            for suffix, algorithm in [("sha1", "sha1"), ("md5", "md5")]:
                f.with_name(f.name + "." + suffix).write_text(
                    hashlib.new(algorithm, f.read_bytes()).hexdigest()
                )
    archive = OUT / "central.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in staging.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(staging).as_posix())
    boundary = uuid.uuid4().hex
    payload = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="bundle"; filename="central.zip"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        ).encode()
        + archive.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    token = base64.b64encode(
        (os.environ[names[0]] + ":" + os.environ[names[1]]).encode()
    ).decode()
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "multipart/form-data; boundary=" + boundary,
    }
    request = urllib.request.Request(
        "https://central.sonatype.com/api/v1/publisher/upload?publishingType=AUTOMATIC",
        data=payload,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=300) as r:
        deployment = r.read().decode().strip().strip('"')
    (OUT / "central-deployment-id.txt").write_text(deployment)
    for _ in range(120):
        req = urllib.request.Request(
            "https://central.sonatype.com/api/v1/publisher/status?id=" + deployment,
            data=b"",
            headers={"Authorization": "Bearer " + token},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            state = json.load(r)
        (OUT / "central-status.json").write_text(json.dumps(state, indent=2))
        if state["deploymentState"] == "PUBLISHED":
            print("Maven Central published:", deployment)
            return
        if state["deploymentState"] == "FAILED":
            raise RuntimeError("Central validation failed; see central-status.json")
        time.sleep(10)
    raise TimeoutError(
        "Central deployment still pending; inspect deployment ID before retrying"
    )


if __name__ == "__main__":
    main()
