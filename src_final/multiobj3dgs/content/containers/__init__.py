"""CPSEG1 framing extracted from tools.content_preparation.packaging.

Wire format stays content-preparation.segment.v1; temporal semantics belong to the
publication profile. Every member range, duplicate, length and hash is verified.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import struct
from ...shared.metadata.serialization import json_bytes, strict_json, sha256_file

MAGIC = b'CPSEG1\n'
SEGMENT_VERSION = 'content-preparation.segment.v1'
MAX_HEADER_BYTES = 16*1024*1024


def write_segment(path, members):
    """Write mapping basename→Path, in sorted order, atomically; return file info."""
    path = Path(path)
    descriptors=[]; offset=0
    for name, source in sorted(members.items()):
        if not name or name in ('.','..') or '/' in name or '\\' in name: raise ValueError('Member must be basename')
        source=Path(source)
        length=source.stat().st_size
        descriptors.append({'id':name,'offset':offset,'length':length,'sha256':sha256_file(source)})
        offset+=length
    encoded=json_bytes({'schema_version':SEGMENT_VERSION,'offset_origin':'data_section','members':descriptors})
    if len(encoded)>MAX_HEADER_BYTES: raise ValueError('Oversized segment header')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    try:
        with temp.open('xb') as target:
            target.write(MAGIC); target.write(struct.pack('<Q',len(encoded))); target.write(encoded)
            for name, source in sorted(members.items()):
                with Path(source).open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''): target.write(block)
            target.flush(); os.fsync(target.fileno())
        # Source changes during streaming must not create a self-inconsistent bundle.
        verify_segment(temp)
        os.replace(temp,path)
    finally: temp.unlink(missing_ok=True)
    return {'bytes':path.stat().st_size,'sha256':sha256_file(path)}


def read_segment_header(path):
    path=Path(path)
    with path.open('rb') as stream:
        if stream.read(len(MAGIC))!=MAGIC: raise ValueError('Invalid segment magic')
        packed=stream.read(8)
        if len(packed)!=8: raise ValueError('Truncated segment header')
        length=struct.unpack('<Q',packed)[0]
        if length>MAX_HEADER_BYTES or length>path.stat().st_size-len(MAGIC)-8: raise ValueError('Invalid segment header size')
        header=strict_json(stream.read(length))
    if header.get('schema_version')!=SEGMENT_VERSION or header.get('offset_origin')!='data_section': raise ValueError('Unsupported CPSEG version')
    offset=0; seen=set(); data_offset=len(MAGIC)+8+length
    for m in header['members']:
        name=m['id']
        if not isinstance(name,str) or not name or name in ('.','..') or '/' in name or '\\' in name or name in seen: raise ValueError('Unsafe/duplicate segment member')
        if any(isinstance(m[k],bool) or not isinstance(m[k],int) for k in ('offset','length')) or m['offset']!=offset or m['length']<0: raise ValueError('Invalid member range')
        if len(m['sha256'])!=64 or any(c not in '0123456789abcdef' for c in m['sha256']): raise ValueError('Invalid member hash')
        offset+=m['length']; seen.add(name)
    if data_offset+offset!=path.stat().st_size: raise ValueError('Truncated/trailing segment bytes')
    return header,data_offset


def extract_segment_member(path, member_id):
    header,start=read_segment_header(path)
    member=next((m for m in header['members'] if m['id']==member_id),None)
    if member is None: raise KeyError(member_id)
    with Path(path).open('rb') as stream:
        stream.seek(start+member['offset']); data=stream.read(member['length'])
    if hashlib.sha256(data).hexdigest()!=member['sha256']: raise ValueError(f'Member hash mismatch: {member_id}')
    return data


def verify_segment(path):
    header,start=read_segment_header(path)
    with Path(path).open('rb') as stream:
        for m in header['members']:
            stream.seek(start+m['offset']); digest=hashlib.sha256(); remaining=m['length']
            while remaining:
                block=stream.read(min(1024*1024,remaining))
                if not block: raise ValueError('Truncated member')
                digest.update(block); remaining-=len(block)
            if digest.hexdigest()!=m['sha256']: raise ValueError(f'Member hash mismatch: {m["id"]}')
    return header
