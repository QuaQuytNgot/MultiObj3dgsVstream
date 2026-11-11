"""Strict project-profile MPD parser. No Torch, CUDA or offline tooling imports."""
from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from uuid import uuid4
import xml.etree.ElementTree as ET
from ...shared.dash import NAMESPACE, PROFILE, INDEX_SCHEME, build_mpd, xml_shape
from ...shared.metadata.serialization import strict_json, relative_path, resolve_path, catalog_from_dict

MAX_MPD_BYTES=4*1024*1024
MAX_INDEX_BYTES=32*1024*1024


def _mpd(data):
    if len(data)>MAX_MPD_BYTES: raise ValueError('MPD exceeds size limit')
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper(): raise ValueError('DTD/entities forbidden')
    root=ET.fromstring(data)
    if root.tag!=f'{{{NAMESPACE}}}MPD' or root.get('profiles')!=PROFILE: raise ValueError('Unsupported MPD profile')
    descriptors=[p for p in root.findall(f'{{{NAMESPACE}}}SupplementalProperty') if p.get('schemeIdUri')==INDEX_SCHEME]
    if len(descriptors)!=1: raise ValueError('Exactly one publication index descriptor required')
    value=descriptors[0].get('value','')
    if value.count('#sha256=')!=1: raise ValueError('Index descriptor must include SHA256')
    path,digest=value.split('#sha256=')
    relative_path(path)
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError('Invalid index digest')
    return root,path,digest


def _construct(root,index_data,index_path,index_digest,base):
    if len(index_data)>MAX_INDEX_BYTES: raise ValueError('Publication index exceeds size limit')
    if hashlib.sha256(index_data).hexdigest()!=index_digest: raise ValueError('Index SHA256 mismatch')
    raw=strict_json(index_data)
    catalog=catalog_from_dict(raw,base)
    expected=ET.fromstring(build_mpd(raw,index_path=index_path,index_sha256=index_digest))
    if xml_shape(root)!=xml_shape(expected): raise ValueError('MPD disagrees with normalized index')
    return catalog


def _read(path,limit):
    with Path(path).open('rb') as stream: data=stream.read(limit+1)
    if len(data)>limit: raise ValueError('Metadata exceeds size limit')
    return data


class MPDParser:
    @staticmethod
    def from_local(path):
        path=Path(path).resolve()
        root,index_path,digest=_mpd(_read(path,MAX_MPD_BYTES))
        sidecar=resolve_path(path.parent,index_path)
        return _construct(root,_read(sidecar,MAX_INDEX_BYTES),index_path,digest,path.parent)

    @staticmethod
    async def from_http(url,client):
        """Fetch via the existing TransferClient; caller owns its lifetime."""
        from ...shared.transfer import RequestSpec
        if urlsplit(url).scheme not in ('http','https'): raise ValueError('Expected HTTP(S) MPD URL')
        async def fetch(target,limit):
            result=await client.fetch(RequestSpec('metadata-'+uuid4().hex,target,metadata={'traffic_kind':'metadata','max_response_bytes':limit}))
            if not result.success or result.body is None: raise ValueError(f'Metadata request failed: {target}: {result.error}')
            if len(result.body)>limit: raise ValueError('Metadata exceeds size limit')
            return result.body
        root,index_path,digest=_mpd(await fetch(url,MAX_MPD_BYTES))
        index_data=await fetch(urljoin(url,index_path),MAX_INDEX_BYTES)
        return _construct(root,index_data,index_path,digest,urljoin(url,'.'))
