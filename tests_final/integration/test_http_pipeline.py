"""Real loopback MPD → verified cache → unchanged codec package handoff."""
import threading
import pytest
from tests_final.content.test_publication import _archive
from multiobj3dgs.content.publication import publish
from multiobj3dgs.client.metadata import MPDParser
from multiobj3dgs.shared.transfer import NetworkClient
from multiobj3dgs.client.cache import ResourceCache
from multiobj3dgs.client.network.observation import TransferObserver
from multiobj3dgs.server.origin import create_server

@pytest.mark.asyncio
async def test_real_http_roundtrip_cache_dedup_and_range(tmp_path):
    archive,originals=_archive(tmp_path)
    root=tmp_path/'published';publish([archive],root,instances=2)
    server=create_server(root,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        async with NetworkClient(http_version='h1') as client:
            catalog=await MPDParser.from_http(url+'/manifest.mpd',client)
            assert len(catalog.instances)==2
            observer=TransferObserver()
            cache=ResourceCache(tmp_path/'cache',1024**2,client,observer=observer)
            content=catalog.contents['example'];a=content.representations['crf-20'].segments[0]
            package=await cache.materialize(a)
            assert (package/'_point_feats.mp4').read_bytes()==originals['gop_000/encoded/crf_20/_point_feats.mp4']
            before=observer.snapshot()['total_bytes'];cache.release_package(package)
            b=content.representations['crf-40'].segments[0]
            package=await cache.materialize(b)
            assert observer.snapshot()['total_bytes']-before==b.resource.bytes
            result=await client.get_range(a.resource.url,0,3)
            assert result.success and result.received_bytes==4 and result.range_honored
            cache.release_package(package);await cache.close()
    finally:
        server.shutdown();thread.join();server.server_close()
