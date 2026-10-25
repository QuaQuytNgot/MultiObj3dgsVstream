import asyncio
import hashlib
from types import SimpleNamespace
import pytest
from multiobj3dgs.client.cache import ResourceCache,CacheBudgetError

def ref(path,expected=None):
    data=path.read_bytes()
    return SimpleNamespace(id=path.name,url=str(path),bytes=len(data),sha256=expected or hashlib.sha256(data).hexdigest())

@pytest.mark.asyncio
async def test_cache_checksum_singleflight_eviction_and_pins(tmp_path):
    a=tmp_path/'a';a.write_bytes(b'a'*10)
    b=tmp_path/'b';b.write_bytes(b'b'*10)
    cache=ResourceCache(tmp_path/'cache',10)
    aa,bb=await asyncio.gather(cache.obtain(ref(a)),cache.obtain(ref(a)))
    assert aa==bb and cache.used_bytes==10
    cache._pins[ref(a).sha256]+=1
    with pytest.raises(CacheBudgetError):await cache.obtain(ref(b))
    cache._pins[ref(a).sha256]-=1
    c=await cache.obtain(ref(b))
    assert c.read_bytes()==b'b'*10 and not aa.exists()
    await cache.close()

@pytest.mark.asyncio
async def test_corruption_never_published(tmp_path):
    a=tmp_path/'a';a.write_bytes(b'wrong')
    cache=ResourceCache(tmp_path/'cache',20)
    with pytest.raises(ValueError):await cache.obtain(ref(a,'a'*64))
    assert cache.used_bytes==0 and cache._reserved==0
    assert not list(cache.blobs.glob('*.part'))
    await cache.close()

@pytest.mark.asyncio
async def test_restart_cache_counts_disk_bytes_and_revalidates_tamper(tmp_path):
    a=tmp_path/'a';a.write_bytes(b'hello')
    path=tmp_path/'cache';cache=ResourceCache(path,10)
    blob=await cache.obtain(ref(a));await cache.close()
    blob.write_bytes(b'wrong')
    cache=ResourceCache(path,10)
    assert cache.used_bytes==5
    restored=await cache.obtain(ref(a))
    assert restored.read_bytes()==b'hello'
    await cache.close()

@pytest.mark.asyncio
async def test_cancelled_consumer_does_not_cancel_shared_fetch(tmp_path):
    from multiobj3dgs.shared.transfer import TransferResult
    data=b'payload';gate=asyncio.Event()
    class Client:
        calls=0
        async def fetch(self,spec):
            self.calls+=1
            await gate.wait()
            return TransferResult(spec,200,data,None,len(data),0.,0.,1.,0.,1.,1.,56.,56.,'h1',{},True,None,None,0.,'test')
    client=Client();cache=ResourceCache(tmp_path/'cache',100,client)
    resource=SimpleNamespace(id='shared',url='http://test.invalid/blob',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    a=asyncio.create_task(cache.obtain(resource));b=asyncio.create_task(cache.obtain(resource))
    await asyncio.sleep(.01);a.cancel()
    with pytest.raises(asyncio.CancelledError):await a
    gate.set()
    assert (await b).read_bytes()==data and client.calls==1
    assert cache._reserved==0
    await cache.close()
