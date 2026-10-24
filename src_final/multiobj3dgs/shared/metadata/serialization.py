"""Canonical index bytes, integrity and path rules shared by publisher/parser."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit
from . import AssetRef, ContentCatalog, FrameMapping, GopSegment, ObjectContent, RepresentationInfo, SceneInstance

SCHEMA = 'multiobj3dgs.streamstgs.publication.v1'
PROFILE = 'urn:multiobj3dgs:dash:profile:streamstgs-independent-gop:1'
INDEX_SCHEME = 'urn:multiobj3dgs:dash:publication-index:1'


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value): raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def revision(payload):
    return hashlib.sha256(json_bytes({k:v for k,v in payload.items() if k != 'revision'})).hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value or any(c.isspace() for c in value):
        raise ValueError(f'Unsafe publication path: {value!r}')
    path = PurePosixPath(value)
    parsed = urlsplit(value)
    if path.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')) or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or '%' in value:
        raise ValueError(f'Unsafe publication path: {value!r}')
    return value


def resolve_path(base, relative):
    relative_path(relative)
    if urlsplit(str(base)).scheme in ('http','https'):
        return urljoin(str(base), relative)
    root = Path(base).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root): raise ValueError('Publication path escapes root')
    return str(resolved)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def verify_file(path, ref):
    path = Path(path)
    if path.stat().st_size != ref.bytes or sha256_file(path) != ref.sha256:
        raise ValueError(f'Resource bytes/hash mismatch: {ref.id}')
    return path


def catalog_from_dict(data, base=''):
    if data.get('schema') != SCHEMA or data.get('profile') != PROFILE:
        raise ValueError('Unsupported publication profile')
    if data.get('revision') != revision(data): raise ValueError('Publication revision mismatch')
    def asset(record):
        return AssetRef(record['id'], resolve_path(base, record['url']), record['bytes'], record['sha256'])
    contents = {}
    resources = {}
    segment_ids = set()
    for obj in data['contents']:
        reps = {}
        for rep in obj['representations']:
            segments = []
            for seg in rep['segments']:
                if seg['id'] in segment_ids: raise ValueError('Duplicate segment ID')
                segment_ids.add(seg['id'])
                segment = GopSegment(seg['id'],seg['gop_index'],seg['start'],seg['duration'],asset(seg['resource']),seg['members'],
                    tuple(asset(a) for a in seg['shared_assets']),tuple(FrameMapping(**f) for f in seg['frames']),seg.get('codec_metadata','reconstruction.json'))
                for resource in segment.resources:
                    prior = resources.setdefault(resource.id, resource)
                    if prior != resource: raise ValueError('Conflicting resource identity')
                segments.append(segment)
            representation = RepresentationInfo(rep['id'],rep['crf'],rep['rank'],rep['bandwidth'],tuple(segments))
            if representation.bandwidth != math.ceil(max(s.cold_bytes*8/s.duration for s in segments)):
                raise ValueError('Bandwidth disagrees with cold required-byte closure')
            if rep['id'] in reps: raise ValueError('Duplicate representation ID')
            reps[rep['id']] = representation
        content = ObjectContent(obj['id'],obj['fps'],obj['duration'],reps,tuple(obj.get('pivot',[0,0,0])),obj.get('codec','streamstgs'))
        if content.id in contents: raise ValueError('Duplicate content ID')
        contents[content.id] = content
    return ContentCatalog(data['revision'],contents,tuple(SceneInstance(**v) for v in data['instances']),data['duration'],str(base))
