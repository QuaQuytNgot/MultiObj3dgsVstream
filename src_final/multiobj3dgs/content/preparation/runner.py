"""Hash-journaled stage execution extracted from the existing preparation workflow.

Stages supply argv lists rather than shell text. The runner never fabricates a
training recipe or imports an upstream training tree into the playback process.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from ...shared.metadata.serialization import json_bytes, strict_json, sha256_file


def _digest(path):
    path=Path(path)
    if path.is_symlink(): raise ValueError(f'Hash inputs/outputs must not be symlinks: {path}')
    if path.is_file(): return sha256_file(path)
    if not path.is_dir(): raise FileNotFoundError(path)
    entries={}
    for item in sorted(path.rglob('*')):
        if item.is_symlink(): raise ValueError(f'Symlink in stage artifact: {item}')
        if item.is_file(): entries[item.relative_to(path).as_posix()]=sha256_file(item)
    if not entries: raise ValueError(f'Empty stage artifact directory: {path}')
    return hashlib.sha256(json_bytes(entries)).hexdigest()


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(json_bytes(value));stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


@dataclass(frozen=True)
class StageSpec:
    name: str
    argv: tuple[str,...]
    cwd: Path
    inputs: tuple[Path,...]
    outputs: tuple[Path,...]
    env: Mapping[str,str] = field(default_factory=dict)
    pinned_inputs: Mapping[str,str] = field(default_factory=dict)
    training: bool = False
    def __post_init__(self):
        if not self.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.name):raise ValueError('Unsafe stage name')
        object.__setattr__(self,'argv',tuple(self.argv))
        if not self.argv or any(not isinstance(v,str) or not v or '\x00' in v for v in self.argv):raise ValueError('Stage command requires nonempty argv strings')
        object.__setattr__(self,'cwd',Path(self.cwd).resolve())
        object.__setattr__(self,'inputs',tuple(Path(p).resolve() for p in self.inputs))
        object.__setattr__(self,'outputs',tuple(Path(p).resolve() for p in self.outputs))
        object.__setattr__(self,'env',MappingProxyType(dict(self.env)))
        object.__setattr__(self,'pinned_inputs',MappingProxyType({str(Path(p).resolve()):v for p,v in self.pinned_inputs.items()}))
        if not self.outputs or len(set(self.outputs))!=len(self.outputs): raise ValueError('Stage needs unique declared outputs')
        if self.training and not self.pinned_inputs: raise ValueError('Training stage requires pinned source inputs')
        if any(not isinstance(k,str) or not isinstance(v,str) for k,v in self.env.items()): raise ValueError('Environment values must be strings')
        if self.training:
            # Pin the script actually invoked, rather than an unrelated file.
            candidates={str((self.cwd/value).resolve()) for value in self.argv if value.endswith('.py')}
            if not candidates.intersection(self.pinned_inputs):raise ValueError('Pin the invoked training Python entrypoint')
    @classmethod
    def from_dict(cls,record,base):
        base=Path(base).resolve()
        def path(raw):return (base/raw).resolve()
        argv=record['argv']
        if not isinstance(argv,list):raise ValueError('argv must be a list, never shell text')
        return cls(record['name'],tuple(argv),path(record.get('cwd','.')),tuple(path(p) for p in record.get('inputs',[])),
            tuple(path(p) for p in record['outputs']),record.get('env',{}),{str(path(p)):v for p,v in record.get('pinned_inputs',{}).items()},record.get('training',False))


class StageRunner:
    def __init__(self,journal, *, resume=False):
        self.path=Path(journal).resolve();self.resume=resume
    def _prepare(self,spec, *, dry_run=False):
        if not spec.cwd.is_dir():raise FileNotFoundError(spec.cwd)
        executable=spec.argv[0]
        if '/' in executable:resolved=Path(executable) if Path(executable).is_absolute() else spec.cwd/executable
        else:
            located=shutil.which(executable,path=spec.env.get('PATH',os.environ.get('PATH','')))
            if located is None:raise FileNotFoundError(f'Stage executable unavailable: {executable}')
            resolved=Path(located)
        pins={}
        for name,expected in spec.pinned_inputs.items():
            actual=_digest(name)
            if actual!=expected:raise ValueError(f'Pinned source checksum mismatch: {name}')
            pins[name]=actual
        inputs={str(p):(_digest(p) if p.exists() or not dry_run else 'pending-upstream-stage') for p in spec.inputs}
        # Include the resolved interpreter/binary so environment changes invalidate resume.
        command={'argv':list(spec.argv),'cwd':str(spec.cwd),'env_sha256':hashlib.sha256(json_bytes(dict(spec.env))).hexdigest(),
            'executable':str(resolved.resolve()),'executable_sha256':_digest(resolved.resolve()),'pinned_inputs':pins,'input_hashes':inputs,
            'outputs':[str(p) for p in spec.outputs],'training':spec.training}
        return command,hashlib.sha256(json_bytes(command)).hexdigest()
    def run(self,spec, *, dry_run=False):
        command,fingerprint=self._prepare(spec,dry_run=dry_run)
        if dry_run:return {'name':spec.name,'status':'dry_run','fingerprint':fingerprint,'command':command}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('Another preparation process owns this journal') from exc
            journal=strict_json(self.path.read_bytes()) if self.path.exists() else {'schema':'multiobj3dgs.preparation-journal.v1','stages':{}}
            if journal.get('schema')!='multiobj3dgs.preparation-journal.v1':raise ValueError('Unsupported preparation journal')
            previous=journal['stages'].get(spec.name)
            if previous and self.resume and previous['status']=='complete' and previous['fingerprint']==fingerprint:
                if all(p.exists() and _digest(p)==previous['output_hashes'].get(str(p)) for p in spec.outputs):
                    return dict(previous,status='reused')
            # Even a failed child may have produced partial output. Require a new
            # stage output location rather than overwrite artifacts implicitly.
            if any(p.exists() for p in spec.outputs):raise FileExistsError(f'Stage outputs already exist: {spec.name}; use matching --resume or new paths')
            entry={'name':spec.name,'status':'running','fingerprint':fingerprint,'command':command,'started_monotonic':time.monotonic()}
            journal['stages'][spec.name]=entry;_atomic_json(self.path,journal)
            try:
                for output in spec.outputs:output.parent.mkdir(parents=True,exist_ok=True)
                log=self.path.parent/(spec.name+'-'+fingerprint[:12]+'-'+str(time.time_ns())+'.log')
                with log.open('xb') as stream:
                    result=subprocess.run(list(spec.argv),cwd=spec.cwd,env=dict(os.environ,**spec.env),stdout=stream,stderr=subprocess.STDOUT,shell=False)
                entry['returncode']=result.returncode;entry['log']=str(log)
                if result.returncode:raise RuntimeError(f'Stage {spec.name} failed with exit {result.returncode}; see {log}')
                # Recheck inputs to catch a stage mutating its declared source.
                if {str(p):_digest(p) for p in spec.inputs}!=command['input_hashes']:raise ValueError('Stage mutated declared source inputs')
                if any(_digest(p)!=digest for p,digest in spec.pinned_inputs.items()):raise ValueError('Stage mutated pinned source inputs')
                entry['output_hashes']={str(p):_digest(p) for p in spec.outputs}
                entry['status']='complete'
            except BaseException as exc:
                entry.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                entry['ended_monotonic']=time.monotonic();_atomic_json(self.path,journal)
            return dict(entry)
