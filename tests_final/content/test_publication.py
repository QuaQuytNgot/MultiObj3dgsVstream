"""Content boundary tests: archive integrity, switching, dedup and wire agreement."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from multiobj3dgs.client.metadata import MPDParser
from multiobj3dgs.content.containers import extract_segment_member, read_segment_header
from multiobj3dgs.content.publication import publish, validate_publication
from multiobj3dgs.content.streamstgs import materialize_gop, validate_archive
from multiobj3dgs.shared.metadata.serialization import json_bytes


def _archive(tmp_path, *, traversal=False, symlink=False, corrupt=False):
    files={'README_STREAMING.txt':b'original readme','video_summary.json':b'{"metadata_only":true}'}
    for gop in range(2):
        for crf in (20,40):
            prefix=f'gop_{gop:03d}/encoded/crf_{crf}/'
            reconstruction={'crf':crf,'segment':{'gop_index':gop,'fps':2,'output_frames':[10+2*gop,11+2*gop],
                'lookahead_frames':[12+2*gop,13+2*gop],
                'timestamps':[{'frame':10+2*gop+i,'seconds':gop+i/2,'model_time':i/4} for i in range(2)]}}
            files.update({prefix+'reconstruction.json':json_bytes(reconstruction),prefix+'manifest.json':json_bytes({'crf':crf}),
                prefix+'_point_feats.mp4':b'hevc'+bytes([crf,gop]),prefix+'_xyz.jxl':b'canonical'+bytes([gop]),
                prefix+'mlp_color.pth':b'common-model'})
    manifest={'schema':1,'object':'example','fps':2,'duration_seconds':2.,'gop_count':2,'crfs':[20,40],'source_frames':[10,13],
        'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()],
        'file_count':len(files),'total_file_bytes':sum(map(len,files.values())),'video_summary':'video_summary.json'}
    files['stream_package_manifest.json']=json_bytes(manifest)
    if corrupt: files['README_STREAMING.txt']=b'Xriginal readme'
    archive=tmp_path/'example.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for name,data in files.items():
            item=tarfile.TarInfo('package/'+name);item.size=len(data)
            stream.addfile(item,io.BytesIO(data))
        if traversal:
            item=tarfile.TarInfo('package/../../escape'); item.size=3;stream.addfile(item,io.BytesIO(b'bad'))
        if symlink:
            item=tarfile.TarInfo('package/link');item.type=tarfile.SYMTYPE;item.linkname='/etc/passwd';stream.addfile(item)
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    return archive,files


@pytest.fixture
def publication(tmp_path):
    archive,originals=_archive(tmp_path)
    root=tmp_path/'published'
    catalog=publish([archive],root,instances=4)
    return archive,originals,root,catalog


def test_realistic_archive_deduplicates_assets_and_keeps_independent_representations(publication):
    _,_,root,catalog=publication
    assert len(catalog.instances)==4
    content=catalog.contents['example']
    assert set(content.representations)=={'crf-20','crf-40'}
    first=content.representations['crf-20'].segments[0]
    other=content.representations['crf-40'].segments[0]
    assert first.shared_assets==other.shared_assets
    assert first.members['_xyz.jxl'] and first.members['_point_feats.mp4']==''
    assert first.cold_bytes==first.resource.bytes+sum(a.bytes for a in first.shared_assets)
    assert first.frame_at(.5).source_frame==11
    assert first.frame_at(.5).model_time==.25
    assert content.segment_at('crf-40',1.).gop_index==1
    with pytest.raises(ValueError): content.segment_at('crf-20',2.)
    assert list(root.rglob('*.cpseg'))
    with pytest.raises(TypeError): content.representations['extra']=None
    with pytest.raises(TypeError): first.members['extra']=''


def test_materializer_recovers_exact_original_codec_bytes(publication,tmp_path):
    _,originals,_,catalog=publication
    for rep in catalog.contents['example'].representations.values():
        for seg in rep.segments:
            destination=tmp_path/f'{rep.id}-{seg.gop_index}'
            materialize_gop(seg,{a.id:Path(a.url) for a in seg.resources},destination)
            prefix=f'gop_{seg.gop_index:03d}/encoded/crf_{rep.crf}/'
            assert {p.name:p.read_bytes() for p in destination.iterdir()}=={k[len(prefix):]:v for k,v in originals.items() if k.startswith(prefix)}
    with pytest.raises(FileExistsError): materialize_gop(seg,{a.id:Path(a.url) for a in seg.resources},destination)


def test_publication_is_deterministic_idempotent_and_immutable(publication,tmp_path):
    archive,_,root,catalog=publication
    original=(root/'manifest.mpd').read_bytes(),(root/'index.json').read_bytes()
    assert publish([archive],root,instances=4).revision==catalog.revision
    assert ((root/'manifest.mpd').read_bytes(),(root/'index.json').read_bytes())==original
    with pytest.raises(FileExistsError): publish([archive],root,instances=2)
    assert ((root/'manifest.mpd').read_bytes(),(root/'index.json').read_bytes())==original
    alternate=tmp_path/'another'
    publish([archive],alternate,instances=4)
    assert (alternate/'manifest.mpd').read_bytes()==original[0]
    assert (alternate/'index.json').read_bytes()==original[1]


@pytest.mark.parametrize('option',['traversal','symlink','corrupt'])
def test_archive_refuses_unsafe_or_unverified_content(tmp_path,option):
    archive,_=_archive(tmp_path,**{option:True})
    with pytest.raises(ValueError): publish([archive],tmp_path/'published')
    assert not (tmp_path/'published').exists()
    assert not (tmp_path.parent/'escape').exists()


def test_archive_checksum_and_all_manifest_files_are_required(tmp_path):
    archive,_=_archive(tmp_path)
    audit=validate_archive(archive)
    assert audit.file_count==22
    Path(str(archive)+'.sha256').write_text('0'*64+'  '+archive.name)
    with pytest.raises(ValueError,match='checksum'): validate_archive(archive)


def test_parser_rejects_mpd_timeline_disagreement(publication):
    _,_,root,_=publication
    data=(root/'manifest.mpd').read_bytes().replace(b'd="2"',b'd="3"',1)
    (root/'manifest.mpd').write_bytes(data)
    with pytest.raises(ValueError,match='disagrees'): MPDParser.from_local(root/'manifest.mpd')


def test_parser_rejects_sidecar_hash_mismatch(publication):
    _,_,root,_=publication
    data=(root/'index.json').read_bytes().replace(b'"duration":2.0',b'"duration":3.0',1)
    (root/'index.json').write_bytes(data)
    with pytest.raises(ValueError,match='SHA256'): MPDParser.from_local(root/'manifest.mpd')


def test_parser_rejects_metadata_traversal_and_dtd(publication):
    _,_,root,_=publication
    original=(root/'manifest.mpd').read_bytes()
    (root/'manifest.mpd').write_bytes(original.replace(b'index.json#',b'../index.json#'))
    with pytest.raises(ValueError,match='Unsafe'): MPDParser.from_local(root/'manifest.mpd')
    (root/'manifest.mpd').write_bytes(b'<!DOCTYPE MPD [<!ENTITY x "boom">]>'+original)
    with pytest.raises(ValueError,match='DTD'): MPDParser.from_local(root/'manifest.mpd')


def test_corrupt_resources_never_materialize_or_validate(publication,tmp_path):
    _,_,root,catalog=publication
    segment=catalog.contents['example'].representations['crf-20'].segments[0]
    resource=Path(segment.resource.url)
    data=bytearray(resource.read_bytes());data[-1]^=1;resource.write_bytes(data)
    with pytest.raises(ValueError,match='hash'): validate_publication(root)
    with pytest.raises(ValueError,match='hash'): materialize_gop(segment,{a.id:Path(a.url) for a in segment.resources},tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_container_rejects_trailing_bytes(publication):
    _,_,_,catalog=publication
    segment=catalog.contents['example'].representations['crf-20'].segments[0]
    path=Path(segment.resource.url)
    header,offset=read_segment_header(path)
    assert extract_segment_member(path,'_point_feats.mp4').startswith(b'hevc')
    with path.open('ab') as stream:stream.write(b'extra')
    with pytest.raises(ValueError,match='trailing'):read_segment_header(path)


def test_http_parser_uses_transport_and_resolves_resource_urls(publication):
    _,_,root,catalog=publication
    class Client:
        def __init__(self):self.specs=[]
        async def fetch(self,spec):
            self.specs.append(spec)
            return SimpleNamespace(success=True,body=(root/spec.url.rsplit('/',1)[1]).read_bytes(),error=None)
    client=Client()
    remote=asyncio.run(MPDParser.from_http('https://example.test/media/manifest.mpd',client))
    assert remote.revision==catalog.revision
    assert len(client.specs)==2
    assert all(s.metadata['traffic_kind']=='metadata' for s in client.specs)
    segment=remote.contents['example'].representations['crf-40'].segments[0]
    assert all(a.url.startswith('https://example.test/media/') for a in segment.resources)


def test_explicit_scene_keeps_independent_offsets_and_transforms(tmp_path):
    archive,_=_archive(tmp_path)
    scene={'duration':3.,'pivots':{'example':[.2,.3,.4]},'instances':[
        {'id':'a','content_id':'example','scene_start':0.,'active_duration':3.,'loop':True},
        {'id':'b','content_id':'example','scene_start':1.,'media_offset':1.,'translation':[1.,2.,3.],'scale':.5}]}
    catalog=publish([archive],tmp_path/'scene',scene=scene)
    assert catalog.contents['example'].pivot==(.2,.3,.4)
    assert catalog.instances[1].media_offset==1.
    assert catalog.instances[1].scale==.5


def test_generated_mpd_passes_pinned_schema_when_xmllint_available(publication):
    import shutil
    from multiobj3dgs.content.publication.schema import validate_mpd_schema
    if shutil.which('xmllint') is None:
        pytest.skip('Optional offline schema check requires xmllint')
    assert validate_mpd_schema(publication[2]/'manifest.mpd')


def _rewrite_index(root,change):
    from multiobj3dgs.shared.metadata.serialization import revision
    from multiobj3dgs.shared.dash import build_mpd
    index=json.loads((root/'index.json').read_bytes());change(index)
    index['revision']=revision(index);data=json_bytes(index)
    (root/'index.json').write_bytes(data)
    (root/'manifest.mpd').write_bytes(build_mpd(index,index_sha256=hashlib.sha256(data).hexdigest()))


def test_validator_compares_normalized_mapping_with_original_codec_metadata(publication):
    _,_,root,_=publication
    def change(index):
        for rep in index['contents'][0]['representations']:
            rep['segments'][0]['frames'][1]['model_time']=.2
    _rewrite_index(root,change)
    MPDParser.from_local(root/'manifest.mpd')  # Self-consistent public metadata.
    with pytest.raises(ValueError,match='Codec metadata disagrees'):validate_publication(root)


def test_parser_checks_actual_closure_bandwidth_and_safe_resource_paths(publication):
    _,_,root,_=publication
    _rewrite_index(root,lambda index:index['contents'][0]['representations'][0].update(bandwidth=1))
    with pytest.raises(ValueError,match='Bandwidth'):MPDParser.from_local(root/'manifest.mpd')
