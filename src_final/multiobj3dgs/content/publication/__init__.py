"""Atomic immutable publication and integrity validation."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import re
from ...shared.metadata.serialization import json_bytes, strict_json, revision, sha256_file, catalog_from_dict, relative_path, verify_file
from ...shared.dash import build_mpd, PROFILE
from ...shared.metadata.serialization import SCHEMA
from ..containers import write_segment, verify_segment, extract_segment_member
from ..streamstgs.archive import extract_verified


def _asset(root,path):
    path=Path(path)
    return {'id':sha256_file(path),'url':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path)}


def _shared(root,source):
    digest=sha256_file(source)
    destination=root/'assets'/digest[:2]/digest
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists(): shutil.copyfile(source,destination)
    return _asset(root,destination)


def _import(root,archive,work):
    manifest,audit=extract_verified(archive,work)
    content_id=manifest['object']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',content_id): raise ValueError('Unsafe content identifier')
    if manifest.get('schema')!=1: raise ValueError('Unsupported source package schema')
    fps=manifest['fps']; duration=manifest['duration_seconds']
    records={r['path']:r for r in manifest['files']}
    # Discover packages from actual metadata, not archive folder naming conventions.
    groups={}
    for relative in sorted(records):
        if not relative.endswith('/reconstruction.json'): continue
        path=work/relative; metadata=strict_json(path.read_bytes())
        segment=metadata['segment']; crf=metadata['crf']; gop=segment['gop_index']
        if (gop,crf) in groups: raise ValueError('Duplicate GOP/CRF metadata')
        if segment['fps']!=fps: raise ValueError('Inconsistent source FPS')
        directory=path.parent
        members={p.name:p for p in directory.iterdir() if p.is_file()}
        if any(p.is_dir() for p in directory.iterdir()): raise ValueError('Nested codec package members unsupported')
        for mandatory in ('_point_feats.mp4','reconstruction.json','manifest.json'):
            if mandatory not in members: raise ValueError(f'Missing codec member: {mandatory}')
        frames=[{'source_frame':entry['frame'],'seconds':entry['seconds'],'model_time':entry['model_time']} for entry in segment['timestamps']]
        if [f['source_frame'] for f in frames]!=segment['output_frames']: raise ValueError('Output frames differ from timestamps')
        groups[gop,crf]=(members,frames)
    gops=list(range(manifest['gop_count'])); crfs=sorted(manifest['crfs'])
    if not crfs or len(set(crfs))!=len(crfs) or set(groups)!={(g,c) for g in gops for c in crfs}: raise ValueError('Source GOP/CRF matrix incomplete')
    reps=[]
    for rank,crf in enumerate(reversed(crfs)):
        segments=[]; representation_id=f'crf-{crf}'
        for gop in gops:
            members,frames=groups[gop,crf]
            common=set.intersection(*(set(groups[gop,c][0]) for c in crfs))
            shared_names={name for name in common if len({sha256_file(groups[gop,c][0][name]) for c in crfs})==1}
            shared_names-={'_point_feats.mp4','reconstruction.json','manifest.json'}
            shared_assets={}; member_refs={}; bundled={}
            for name,path in sorted(members.items()):
                if name in shared_names:
                    asset=_shared(root,path); shared_assets[asset['id']]=asset; member_refs[name]=asset['id']
                else:
                    bundled[name]=path; member_refs[name]=''
            target=root/'segments'/content_id/representation_id/f'gop-{gop:05d}.cpseg'
            write_segment(target,bundled)
            start=frames[0]['seconds']; segment_duration=len(frames)/fps
            segments.append({'id':f'{content_id}:{representation_id}:{gop}','gop_index':gop,'start':start,'duration':segment_duration,
                'resource':_asset(root,target),'members':member_refs,'shared_assets':sorted(shared_assets.values(),key=lambda a:a['id']),
                'frames':frames,'codec_metadata':'reconstruction.json'})
        bandwidth=math.ceil(max(sum({a['sha256']:a['bytes'] for a in [s['resource'],*s['shared_assets']]}.values())*8/s['duration'] for s in segments))
        reps.append({'id':representation_id,'crf':crf,'rank':rank,'bandwidth':bandwidth,'segments':segments})
    reps.sort(key=lambda r:r['id'])
    all_frames=[f['source_frame'] for s in reps[0]['segments'] for f in s['frames']]
    begin,end=manifest['source_frames']
    if all_frames!=list(range(begin,end+1)): raise ValueError('Source frame coverage mismatch')
    # Original package index and non-codec files remain available as provenance,
    # without turning them into playback dependencies or altering their contents.
    provenance=[]
    for name in ('stream_package_manifest.json',manifest.get('video_summary','video_summary.json'),'README_STREAMING.txt'):
        path=work/relative_path(name)
        if path.is_file(): provenance.append(_shared(root,path))
    content={'id':content_id,'codec':'streamstgs','fps':fps,'duration':duration,'pivot':[0.,0.,0.],'representations':reps}
    return content,{'archive_sha256':audit.sha256,'content_id':content_id,'verified_file_count':audit.file_count,'verified_source_bytes':audit.verified_bytes,'original_metadata':provenance}


def validate_publication(path, *, verify_resources=True):
    from ...client.metadata import MPDParser
    path=Path(path)
    catalog=MPDParser.from_local(path/'manifest.mpd' if path.is_dir() else path)
    if verify_resources:
        seen=set()
        for content in catalog.contents.values():
            for rep in content.representations.values():
                for segment in rep.segments:
                    for ref in segment.resources:
                        if ref.sha256 not in seen: verify_file(ref.url,ref); seen.add(ref.sha256)
                    header=verify_segment(segment.resource.url)
                    expected={k for k,v in segment.members.items() if not v}
                    if {m['id'] for m in header['members']}!=expected: raise ValueError('Bundle/index member mismatch')
                    codec=strict_json(extract_segment_member(segment.resource.url,segment.codec_metadata))
                    mapping=codec['segment']
                    expected_frames=[{'frame':f.source_frame,'seconds':f.seconds,'model_time':f.model_time} for f in segment.frames]
                    if codec['crf']!=rep.crf or mapping['gop_index']!=segment.gop_index or mapping['fps']!=content.fps or mapping['timestamps']!=expected_frames or mapping['output_frames']!=[f.source_frame for f in segment.frames]:
                        raise ValueError('Codec metadata disagrees with publication GOP/frame mapping')
                    original=strict_json(extract_segment_member(segment.resource.url,'manifest.json'))
                    if 'files' in original:
                        actual={m['id']:{'bytes':m['length'],'sha256':m['sha256']} for m in header['members'] if m['id']!='manifest.json'}
                        shared={a.id:a for a in segment.shared_assets}
                        actual.update({name:{'bytes':shared[asset_id].bytes,'sha256':shared[asset_id].sha256} for name,asset_id in segment.members.items() if asset_id})
                        if original['files']!=actual or original['actual_payload_bytes']!=sum(v['bytes'] for v in actual.values()):
                            raise ValueError('Original codec manifest disagrees with packaged member bytes')
        index_path=Path(catalog.base_url)/'index.json'
        raw=strict_json(index_path.read_bytes())
        for record in raw.get('provenance',[]):
            for ref in record['original_metadata']:
                from ...shared.metadata import AssetRef
                verify_file(Path(catalog.base_url)/relative_path(ref['url']),AssetRef(**ref))
    return catalog


def publish(archives, output, *, instances=None, scene=None):
    """Create a new publication, or verify and reuse an identical immutable one.

    An exception leaves the previous publication intact. Retries recreate staging
    deterministically; no incomplete publication is visible at the final path.
    """
    archives=[Path(a) for a in archives]
    if not archives: raise ValueError('At least one archive required')
    if instances is not None and (isinstance(instances,bool) or not isinstance(instances,int) or instances<=0): raise ValueError('instances must be positive')
    if scene is not None and instances is not None: raise ValueError('Choose scene or instances, not both')
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        contents=[]; provenance=[]
        for archive in sorted(archives,key=lambda p:p.name):
            with tempfile.TemporaryDirectory(prefix='multiobj3dgs-import-') as temporary:
                content,origin=_import(staging,archive,Path(temporary))
                if any(c['id']==content['id'] for c in contents): raise ValueError('Duplicate content identity')
                contents.append(content); provenance.append(origin)
        contents.sort(key=lambda c:c['id'])
        if scene is None:
            count=instances if instances is not None else len(contents)
            configured=[{'id':f'instance-{i}','content_id':contents[i%len(contents)]['id'],'translation':[(i-(count-1)/2)*1.8,0.,0.]} for i in range(count)]
            scene_duration=max(c['duration'] for c in contents)
        else:
            configured=scene['instances']; scene_duration=scene['duration']
            for content in contents:
                if content['id'] in scene.get('pivots',{}): content['pivot']=scene['pivots'][content['id']]
        index={'schema':SCHEMA,'profile':PROFILE,'contents':contents,'instances':configured,'duration':scene_duration,
            'provenance':provenance,'bandwidth_policy':'ceil(peak cold unique required resource bytes * 8 / GOP duration)',
            'switching_policy':'independent representations; GOP boundaries only',
            'interoperability':'CPSEG + project StreamSTGS codec adapter; not generic HEVC DASH playback'}
        index['revision']=revision(index)
        catalog_from_dict(index,staging)
        encoded=json_bytes(index); digest=hashlib.sha256(encoded).hexdigest()
        (staging/'index.json').write_bytes(encoded)
        (staging/'manifest.mpd').write_bytes(build_mpd(index,index_sha256=digest))
        validate_publication(staging)
        if output.exists():
            existing=validate_publication(output)
            if existing.revision!=index['revision']: raise FileExistsError('Immutable publication differs; choose a new output directory')
            return existing
        os.rename(staging,output)
        return validate_publication(output,verify_resources=False)
    finally:
        if staging.exists(): shutil.rmtree(staging)
