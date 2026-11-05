# Third-party licenses and attribution

Only third-party copyright, license and source-credit notices belong here.
Implementation status, algorithm descriptions, test counts and migration logs
are not release notices and are excluded.

`MIGRATION-NOTICES.txt` and `migration-notices.json` contain reviewed source
attribution. The byte offsets and hashes protect the retained original text.
Generated Rust copies are synchronized with:

```powershell
python scripts/migration/sync_migration_notices.py --check
```

Release packages share complete identical license terms and retain each
component's copyright and any additional conditions. Their index reconstructs
the exact original text for verification. Modified or unknown terms are kept
verbatim, never replaced based only on the license name.

After changing dependencies or reviewed source attribution, regenerate:

```powershell
python scripts/migration/generate_release_notice_policy.py
```

The normal build/staging scripts apply this policy automatically.
