#!/usr/bin/env python3
"""Fetch pinned official prebuilt viewer modules, or verify the local copies.

No Node install or build is required. Archives and extracted files are checked
against the reviewed lock before any destination file is replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def vendor(lock_path=ROOT / "dependencies.lock.json", destination=None, verify=False):
    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf8"))
    if lock.get("schema") != "content-preparation.viewer-dependencies.v1":
        raise ValueError("Unsupported viewer dependency lock")
    destination = Path(destination or lock_path.parent / "vendor").resolve()
    outputs = []
    for package in lock["packages"]:
        files = []
        missing = False
        for row in package["files"]:
            target = (destination / row["path"]).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Viewer dependency path escapes destination")
            valid = target.is_file() and digest(target.read_bytes()) == row["sha256"]
            if verify and not valid:
                raise ValueError(f"Missing or changed viewer dependency: {target}")
            files.append((row, target))
            missing |= not valid
        if missing:
            request = urllib.request.Request(package["archive_url"], headers={"User-Agent": "MultiObj3dgsVstream-content-viewer/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                archive_bytes = response.read()
            if digest(archive_bytes) != package["archive_sha256"]:
                raise ValueError(f"Archive checksum mismatch for {package['name']}")
            pending = []
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                for row, target in files:
                    member = archive.getmember("package/" + row["archive_path"])
                    if not member.isfile():
                        raise ValueError("Expected a regular archive file")
                    data = archive.extractfile(member).read()
                    if digest(data) != row["sha256"]:
                        raise ValueError(f"Dependency checksum mismatch for {row['path']}")
                    pending.append((target, data))
            for target, data in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(target)
        outputs.extend({"path": str(target.relative_to(destination)), "bytes": target.stat().st_size,
                        "sha256": row["sha256"]} for row, target in files)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "dependencies.lock.json")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify", action="store_true", help="Verify local pinned files without network access")
    args = parser.parse_args(argv)
    files = vendor(args.lock, args.destination, args.verify)
    print(json.dumps({"status": "verified", "files": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
