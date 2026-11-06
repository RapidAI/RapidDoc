"""Apply reviewed release-only exclusions; keep source audit records separately."""
import json
from pathlib import Path

from license_bundle import read_tree, sha

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT/'packaging/release-notice-policy.json'
CARGO_NOTICES = ROOT/'packaging/release-cargo-notices'
DEVELOPMENT_FILES = {
    'migration-notices.json', 'office/office-native-provenance.json',
    'office/office-image-dependency-notices.json', 'office/office-chart-dependency-notices.json',
}
REPLACED_FILES = {'MODEL-DOWNLOAD-NOTICES.txt', 'FIXED-FONT-LOCK-LICENSES.txt', 'MIGRATION-NOTICES.txt'}


def load_policy():
    if not POLICY.is_file():
        raise ValueError('Generate the release notice policy before packaging')
    policy = json.loads(POLICY.read_text('utf8'))
    if policy['cargo_lock_sha256'] != sha((ROOT/'Cargo.lock').read_bytes().replace(b'\r\n',b'\n')):
        raise ValueError('Cargo.lock changed; regenerate the release notice policy')
    for path,expected in policy.get('manifest_hashes',{}).items():
        if sha((ROOT/path).read_bytes().replace(b'\r\n',b'\n')) != expected:
            raise ValueError(f'{path} changed; regenerate the release notice policy')
    return policy


def filter_entries(entries, kind, profile='all', *, policy=None, generated=None, office=None):
    policy = policy if policy is not None else load_policy()
    selected = set(policy['profiles'][profile])
    result, removed = {}, []
    if kind == 'main':
        entries = dict(entries)
        if generated is None:
            generated = read_tree(CARGO_NOTICES)
        entries.update(generated)
        if office is None:
            office = read_tree(ROOT/'rust/rapiddoc-office/licenses')
        entries.update({'office/'+name: data for name, data in office.items()})
    for name, data in entries.items():
        reason = None
        if kind in ('main', 'office-source'):
            if name in DEVELOPMENT_FILES:
                reason = 'development hashes/test provenance; retained in source packaging records'
            elif name in REPLACED_FILES:
                reason = 'replaced by selected runtime component licenses, without repeated alternatives'
            elif name == 'fonttools-glyph-data/LICENSE.external':
                reason = 'fontTools test fonts/cu2qu are not distributed; Adobe AGL notice is retained'
            else:
                rule = policy['files'].get(('office/'+name) if kind == 'office-source' else name)
                if rule:
                    if rule['component'] not in selected:
                        reason = 'component absent from this target runtime graph'
                    elif not rule['selected']:
                        reason = 'alternative license not selected; retained license satisfies the SPDX expression'
        elif kind == 'pdfium':
            if name.startswith('BUILD_LICENSES/android/') and profile in ('windows', 'linux', 'windows-aarch64', 'linux-aarch64', 'all'):
                reason = 'Android-only build component, not a desktop release target'
            elif name.startswith(('BUILD_LICENSES/pdfium-binaries_musl/', 'BUILD_LICENSES/cibw_musl/')) and profile in ('windows', 'linux', 'windows-aarch64', 'linux-aarch64'):
                reason = 'musl-only component; release is Windows MSVC or Linux glibc/manylinux'
        if reason:
            removed.append(dict(path=name, sha256=sha(data), bytes=len(data), reason=reason))
        else:
            result[name] = data
    return result, removed
