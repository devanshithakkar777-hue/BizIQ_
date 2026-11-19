"""Encryption key operations for BizIQ. See docs/SECURITY.md for the full procedures.

    python scripts/rotate_keys.py status                  # which keys exist, which master key wraps them, what is encrypted
    python scripts/rotate_keys.py encrypt [--org ID]      # turn existing plaintext datasets into encrypted ones (all orgs by default)
    python scripts/rotate_keys.py decrypt [--org ID]      # the reverse, for rollback
    python scripts/rotate_keys.py master                  # re-wrap every data key under a NEW master key (cheap; data is not rewritten)
    python scripts/rotate_keys.py data --org ID           # new data key for one organisation, re-encrypt its datasets, retire the old key

Rotating the master key:  1) generate a new one:  python -c "from biziq import crypto; print(crypto.generate_master_key())"
                          2) set BIZIQ_MASTER_KEY=<new>  and  BIZIQ_MASTER_KEY_PREVIOUS=<old>
                          3) run `master`   4) remove BIZIQ_MASTER_KEY_PREVIOUS.   Back up the new key before step 3.
Losing the master key means the encrypted data cannot be recovered by anyone, including you. Store it in a password manager
or secrets vault, apart from the database backups.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biziq import accounts, crypto, data_layer, db  # noqa: E402


def orgs(only: str | None) -> list[str]:
    ids = accounts.list_org_ids()
    if only:
        if only not in ids:
            raise SystemExit(f"No organisation '{only}'. Known: {', '.join(ids) or 'none'}")
        return [only]
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "encrypt", "decrypt", "master", "data"])
    ap.add_argument("--org")
    a = ap.parse_args(argv)
    if not crypto.available():
        print("No key configured. Set BIZIQ_MASTER_KEY first (generate one with: python -c \"from biziq import crypto; print(crypto.generate_master_key())\").")
        return 2
    if a.cmd == "status":
        print(f"backend: {db.backend_name()}   data encryption flag (BIZIQ_ENCRYPT_DATA): {'on' if data_layer.encrypt_on() else 'off'}")
        for k in crypto.key_status():
            print(f"  {k['scope']:40s} v{k['version']} {'active ' if k['active'] else 'old    '} master:{'current' if k['wrapped_by_current_master'] else 'OLD (run `master`)'}")
        for org in orgs(None):
            accounts.activate_org(org)
            with data_layer._conn() as c:
                tables = [t for t in data_layer.SCHEMAS if db.table_exists(c, t)]
                enc = [t for t in tables if data_layer._is_encrypted_table(c, t)]
            print(f"  org {org}: {len(enc)}/{len(tables)} datasets encrypted")
        return 0
    if a.cmd in ("encrypt", "decrypt"):
        for org in orgs(a.org):
            accounts.activate_org(org)
            res = data_layer.convert_storage(a.cmd == "encrypt")
            print(f"  {org}: {sum(v == 'converted' for v in res.values())} converted, {sum(v == 'already' for v in res.values())} already {a.cmd}ed")
        return 0
    if a.cmd == "master":
        print(f"Re-wrapped {crypto.rewrap_all()} data key(s) under the current master key. You can now remove BIZIQ_MASTER_KEY_PREVIOUS.")
        return 0
    if a.cmd == "data":
        if not a.org:
            raise SystemExit("Give --org ID.")
        accounts.activate_org(a.org)
        scope = data_layer.current_scope()
        kid = crypto.new_key_version(scope)
        n = data_layer.reencrypt_all()
        crypto.retire_old_versions(scope)
        print(f"{a.org}: new data key v{kid}; {n} dataset table(s) re-encrypted; old key retired.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
