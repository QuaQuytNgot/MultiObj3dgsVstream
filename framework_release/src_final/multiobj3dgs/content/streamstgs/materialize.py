"""Verified resource closure -> decoder's unchanged GOP-local filenames."""
from pathlib import Path
import os
import shutil
import tempfile
from ...shared.metadata.serialization import verify_file
from ..containers import read_segment_header, extract_segment_member


def materialize(segment, resource_path, asset_paths, destination):
    resource_path=verify_file(resource_path,segment.resource)
    header,_=read_segment_header(resource_path)
    bundled={m['id'] for m in header['members']}
    expected={name for name,asset_id in segment.members.items() if not asset_id}
    if bundled!=expected: raise ValueError('Bundle members differ from publication index')
    assets={a.id:a for a in segment.shared_assets}
    destination=Path(destination)
    if destination.exists():
        # Never overwrite a decoder package potentially owned by active playback.
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        for name,asset_id in segment.members.items():
            target=temporary/name
            if asset_id:
                source=verify_file(asset_paths[asset_id],assets[asset_id])
                # A decoder may mutate its files; hardlinks would corrupt the cache.
                shutil.copyfile(source,target)
            else:
                target.write_bytes(extract_segment_member(resource_path,name))
        os.rename(temporary,destination)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return destination


def materialize_gop(segment, resource_paths, destination):
    return materialize(segment,resource_paths[segment.resource.id],resource_paths,destination)
