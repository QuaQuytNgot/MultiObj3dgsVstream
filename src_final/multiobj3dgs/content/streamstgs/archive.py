"""Safe, byte-verified import of existing StreamSTGS compressed archives."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import shutil
import tarfile
import tempfile
from ...shared.metadata.serialization import relative_path, sha256_file, strict_json


@dataclass(frozen=True)
class ArchiveAudit:
    path: str
    sha256: str
    object_id: str
    file_count: int
    verified_bytes: int


def extract_verified(archive, destination, *, max_bytes=32*1024**3, require_checksum=True):
    """Extract regular files only; verify archive sidecar and every manifest entry.

    Writes exclusively into a caller-owned empty staging directory. No tarfile
    extraction API is used, so symlink, hardlink and traversal members are rejected.
    """
    archive=Path(archive); destination=Path(destination)
    actual_hash=sha256_file(archive)
    checksum=Path(str(archive)+'.sha256')
    if require_checksum and not checksum.is_file(): raise ValueError(f'Archive checksum sidecar missing: {checksum}')
    if checksum.is_file():
        fields=checksum.read_text().strip().split()
        if len(fields)!=2 or fields[0]!=actual_hash or fields[1].lstrip('*')!=archive.name:
            raise ValueError('Archive checksum sidecar mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()): raise ValueError('Extraction staging directory must be empty')
    with tarfile.open(archive,'r:*') as tar:
        members=tar.getmembers()
        if not members or len(members)>100000: raise ValueError('Invalid archive member count')
        seen=set(); total=0; roots=set()
        for member in members:
            name=member.name.rstrip('/')
            relative_path(name)
            if name in seen: raise ValueError('Duplicate archive member')
            seen.add(name); roots.add(PurePosixPath(name).parts[0])
            if not (member.isdir() or member.isfile()) or member.issym() or member.islnk():
                raise ValueError('Archive links/special members are forbidden')
            if member.size<0: raise ValueError('Negative archive member size')
            total+=member.size
            if total>max_bytes: raise ValueError('Archive uncompressed size exceeds limit')
        if len(roots)!=1: raise ValueError('Archive must have one root directory')
        root=next(iter(roots)); prefix=root+'/'
        manifest_name=prefix+'stream_package_manifest.json'
        try: manifest_member=tar.getmember(manifest_name)
        except KeyError as exc: raise ValueError('Stream package manifest missing') from exc
        if manifest_member.size>32*1024*1024:raise ValueError('Oversized source package manifest')
        manifest=strict_json(tar.extractfile(manifest_member).read())
        declared={}
        for record in manifest['files']:
            name=relative_path(record['path'])
            if name in declared: raise ValueError('Duplicate manifest file')
            if isinstance(record['bytes'],bool) or not isinstance(record['bytes'],int) or record['bytes']<0: raise ValueError('Invalid declared bytes')
            declared[name]=record
        regular={m.name[len(prefix):] for m in members if m.isfile() and m.name!=manifest_name}
        if regular!=set(declared): raise ValueError('Archive file set differs from manifest')
        if manifest.get('file_count')!=len(declared) or manifest.get('total_file_bytes')!=sum(v['bytes'] for v in declared.values()):
            raise ValueError('Manifest file/byte totals mismatch')
        for member in members:
            if not member.isfile(): continue
            name=member.name[len(prefix):]
            target=destination / relative_path(name)
            target.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256(); copied=0
            with tar.extractfile(member) as source,target.open('xb') as output:
                for block in iter(lambda:source.read(1024*1024),b''):
                    output.write(block); digest.update(block); copied+=len(block)
            if name!='stream_package_manifest.json':
                expected=declared[name]
                if copied!=expected['bytes'] or digest.hexdigest()!=expected['sha256']:
                    raise ValueError(f'Original payload bytes/hash mismatch: {name}')
    return manifest,ArchiveAudit(str(archive),actual_hash,manifest['object'],len(declared),sum(v['bytes'] for v in declared.values()))


def validate_archive(archive):
    with tempfile.TemporaryDirectory(prefix='multiobj3dgs-audit-') as temporary:
        _,audit=extract_verified(archive,temporary)
    return audit
