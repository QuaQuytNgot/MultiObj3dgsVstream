"""Verified disk cache with byte reservations, singleflight and package leases."""
from collections import OrderedDict, Counter
import asyncio
import hashlib
from pathlib import Path
import shutil
from urllib.parse import urlsplit
from uuid import uuid4
from multiobj3dgs.shared.transfer import RequestSpec


class CacheBudgetError(MemoryError):
    pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


class ResourceCache:
    """Use on one I/O event loop. release_package may run on the GPU owner.

    Immutable file cache keys are SHA256; decoded instance state is never shared.
    obtain paths remain valid until subsequent admission. materialize pins every
    required blob and its package until release_package, including in-flight jobs.
    """
    def __init__(self,root,budget_bytes,client=None,observer=None,events=None):
        if isinstance(budget_bytes,bool) or not isinstance(budget_bytes,int) or budget_bytes<=0:
            raise ValueError('cache budget must be positive bytes')
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.blobs=self.root/'blobs';self.blobs.mkdir(exist_ok=True)
        self.packages=self.root/'packages';self.packages.mkdir(exist_ok=True)
        self.budget_bytes=budget_bytes;self.client=client
        self.observer=observer;self.events=events
        self._entries=OrderedDict();self._pins=Counter();self._pending={}
        self._reserved=0;self._packages={};self._closed=False
        self._verified=set()
        import threading
        self._guard=threading.RLock()
        # Packages are session-owned. Do not consume/delete another run's leases.
        self.session=self.packages/uuid4().hex;self.session.mkdir()
        # Existing blobs count toward disk budget, but require verification before
        # first reuse in this session (a crash or external edit cannot mint credit).
        for path in self.blobs.iterdir():
            if path.is_file() and len(path.name)==64 and all(c in '0123456789abcdef' for c in path.name):
                self._entries[path.name]=(path,path.stat().st_size)
        self._reserve(0)

    @property
    def used_bytes(self):
        with self._guard:
            return sum(n for _,n in self._entries.values())+sum(v[1] for v in self._packages.values())

    def _reserve(self,n):
        with self._guard:
            for key in list(self._entries):
                if self.used_bytes+self._reserved+n<=self.budget_bytes:break
                if not self._pins[key] and key not in self._pending:
                    p,_=self._entries.pop(key);p.unlink(missing_ok=True)
            if self.used_bytes+self._reserved+n>self.budget_bytes:
                raise CacheBudgetError(f'Cache needs {n} bytes, used={self.used_bytes}, reserved={self._reserved}, budget={self.budget_bytes}')
            self._reserved+=n

    def missing_bytes(self,segment):
        """Union of unverified resources, excluding planned/in-flight cache credit."""
        segments=(segment,) if hasattr(segment,'resource') else tuple(segment)
        unique={r.sha256:r for seg in segments for r in (seg.resource,)+tuple(seg.shared_assets)}
        with self._guard:
            return sum(r.bytes for k,r in unique.items() if k not in self._verified)

    async def obtain(self,ref,traffic_kind="shared_asset"):
        if self._closed:raise RuntimeError('Cache closed')
        key=ref.sha256
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid SHA256')
        if key in self._entries:
            path,size=self._entries[key]
            if key not in self._verified:
                valid=size==ref.bytes and await asyncio.to_thread(digest,path)==key
                if not valid:
                    self._entries.pop(key);path.unlink(missing_ok=True)
                    return await self.obtain(ref,traffic_kind)
                self._verified.add(key)
            if size!=ref.bytes:raise ValueError('Conflicting size for same hash')
            self._entries.move_to_end(key)
            if self.events:self.events.emit('cache_hit',bytes=ref.bytes,asset_id=ref.id)
            return path
        task=self._pending.get(key)
        if task is None:
            self._reserve(ref.bytes)
            task=asyncio.create_task(self._fetch(ref,traffic_kind));self._pending[key]=task
            # Retrieve exceptions even if all consumers cancel their shield.
            task.add_done_callback(lambda f:f.exception() if not f.cancelled() else None)
        return await asyncio.shield(task)

    async def _fetch(self,ref,traffic_kind):
        key=ref.sha256;path=self.blobs/key;temporary=self.blobs/(key+'.'+uuid4().hex+'.part')
        spec=None
        try:
            if path.is_file() and path.stat().st_size==ref.bytes and await asyncio.to_thread(digest,path)==key:
                pass
            elif urlsplit(ref.url).scheme in ('http','https'):
                if self.client is None:raise ValueError('HTTP resources require TransferClient')
                spec=RequestSpec(uuid4().hex,ref.url,metadata={'max_response_bytes':ref.bytes,'traffic_kind':traffic_kind})
                if self.observer:self.observer.started(spec)
                if hasattr(self.client,'download_to_file'):
                    result=await self.client.download_to_file(spec,temporary)
                else:
                    result=await self.client.fetch(spec)
                    if result.success and result.body is not None:
                        temporary.write_bytes(result.body)
                if self.observer:self.observer(result)
                if not result.success:raise IOError(f'Fetch failed {ref.url}: {result.error}')
                if not temporary.exists() or temporary.stat().st_size!=ref.bytes or await asyncio.to_thread(digest,temporary)!=key:
                    raise ValueError(f'Resource size/hash mismatch: {ref.id}')
                temporary.replace(path)
            else:
                source=Path(ref.url)
                if not source.is_file() or source.stat().st_size!=ref.bytes:raise ValueError(f'Resource size mismatch: {ref.id}')
                await asyncio.to_thread(shutil.copyfile,source,temporary)
                if await asyncio.to_thread(digest,temporary)!=key:raise ValueError(f'Resource checksum mismatch: {ref.id}')
                temporary.replace(path)
            with self._guard:
                self._entries[key]=(path,ref.bytes)
                self._verified.add(key)
            return path
        finally:
            temporary.unlink(missing_ok=True)
            if spec is not None and self.observer:self.observer.cancelled(spec)
            with self._guard:self._reserved-=ref.bytes
            self._pending.pop(key,None)

    async def materialize(self,segment):
        from multiobj3dgs.content.streamstgs.materialize import materialize
        refs=(segment.resource,)+tuple(segment.shared_assets)
        keys=tuple(set(r.sha256 for r in refs));paths={}
        with self._guard:
            for key in keys:self._pins[key]+=1
        destination=self.session/uuid4().hex
        reservation=0
        try:
            for ref in refs:paths[ref.id]=await self.obtain(ref, "media" if ref.sha256==segment.resource.sha256 else "shared_asset")
            # Conservative physical duplication estimate: package includes bundle
            # member bytes plus shared files. Actual package bytes replace reserve.
            needed=sum({r.sha256:r.bytes for r in refs}.values())
            self._reserve(needed)
            reservation=needed
            job=asyncio.create_task(asyncio.to_thread(materialize,segment,paths[segment.resource.id],paths,destination))
            try:
                await asyncio.shield(job)
            except asyncio.CancelledError:
                await job
                raise
            size=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
            if size>reservation:raise CacheBudgetError('Materialized package exceeded reservation')
            with self._guard:self._packages[destination]=(keys,size)
            return destination
        except BaseException:
            # A started materializer is drained before deleting its package.
            shutil.rmtree(destination,ignore_errors=True)
            with self._guard:
                for key in keys:self._pins[key]-=1
            raise
        finally:
            with self._guard:self._reserved-=reservation

    def release_package(self,path):
        path=Path(path)
        with self._guard:
            entry=self._packages.pop(path,None)
            if entry:
                for key in entry[0]:self._pins[key]-=1
        if entry:shutil.rmtree(path,ignore_errors=True)

    async def close(self):
        self._closed=True
        await asyncio.gather(*tuple(self._pending.values()),return_exceptions=True)
        for path in list(self._packages):self.release_package(path)
        shutil.rmtree(self.session,ignore_errors=True)
