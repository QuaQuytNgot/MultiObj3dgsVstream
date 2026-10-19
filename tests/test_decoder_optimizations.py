"""Corruption, immutable ownership and exact reconstruction of optimized paths."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from src.client.adaptation_inputs import PayloadDescriptor
from src.client.decoder import ProgressiveDecoder, MissingDependencyError
from src.client.optimized_codec import ByteCache, OptimizedDraco, MemoryDraco
from tools.content_preparation.assets import synthetic_state, state_hash, canonical_json
from tools.content_preparation.codec_adapter import get_codec

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not all((ROOT/"output/build/content_preparation"/name).exists()
            for name in ("draco_byte_codec", "libdraco_memory.so")),
    reason="Build the Draco bridge and memory library")


def packet(tmp_path, frame, quality, target, parent=None, dependency=None):
    target.metadata.update(frame=frame, quality=f"Q{quality}")
    codec = get_codec({"name":"gaussian_attribute_draco_byteplanes"})
    path = tmp_path/f"{frame}-{quality}.cpgs"
    record = codec.encode(target,path,parent=parent)
    d = PayloadDescriptor(f"actor:{frame}:{quality}","actor",frame,quality,0,frame/30,
                          "segment",record["decoded_state_hash"],() if dependency is None else (dependency,),record["parent_state_hash"])
    return d,path.read_bytes(),codec.decode(path,parent)


@pytest.mark.parametrize("native",["subprocess","memory"])
def test_exact_progression_static_reuse_and_immutable_ownership(tmp_path,native):
    decoder = ProgressiveDecoder(optimization={"native":native})
    previous_states = {}
    for frame in (1,2,3):
        parent = None
        for q,n in enumerate((4,7,10,13)):
            target = synthetic_state(1,n)
            target.arrays["xyz"] += np.float32(frame/10)
            d,blob,reference = packet(tmp_path,frame,q,target,parent,None if q==0 else f"actor:{frame}:{q-1}")
            actual = decoder.decode_payload(d,blob).state
            assert state_hash(actual)==state_hash(reference)
            for name in reference.arrays:
                assert actual.arrays[name].tobytes()==reference.arrays[name].tobytes()
            if q in previous_states:
                assert actual.arrays["sh"] is previous_states[q].arrays["sh"]
                assert actual.arrays["xyz"] is not previous_states[q].arrays["xyz"]
            with pytest.raises(ValueError): actual.arrays["sh"].flags.writeable=True
            with pytest.raises(TypeError): actual.arrays["sh"]=np.zeros((n,3,1))
            with pytest.raises(AttributeError): actual.ids=np.arange(n)
            previous_states[q]=actual
            parent=reference
        for q in (0,3,1,2): assert decoder.get_frame("actor",frame,q).quality==q
    assert decoder.optimized.stats["stream_hits"]>0
    assert decoder.optimized.stats["array_hits"]>0
    decoder.close()
    assert decoder.optimized.streams.size==decoder.optimized.arrays.size==0


@pytest.mark.parametrize("native",["subprocess","memory"])
@pytest.mark.parametrize("verification",["strict","runtime"])
def test_reorder_deletion_schema_change_empty_and_random_access(tmp_path,native,verification):
    base = synthetic_state(3,8)
    a,blob,reference=packet(tmp_path,3,0,base)
    target=synthetic_state(3,12).subset([11,1,4,0,10])
    target.arrays["sh"]=np.repeat(target.arrays["sh"],4,axis=2)
    b,enh,expected=packet(tmp_path,3,1,target,reference,a.payload_id)
    decoder=ProgressiveDecoder(optimization={"native":native,"verification":verification})
    def decode(d,data):
        return decoder.decode_payload(d,data,expected_checksum=hashlib.sha256(data).hexdigest())
    with pytest.raises(MissingDependencyError): decode(b,enh)
    decode(a,blob)
    assert state_hash(decode(b,enh).state)==state_hash(expected)
    # Different object needs its own same-frame parent even when bytes repeat.
    other_a=replace(a,payload_id="other:a",object_id="other")
    other_b=replace(b,payload_id="other:b",object_id="other",dependencies=("other:a",))
    with pytest.raises(MissingDependencyError): decode(other_b,enh)
    decode(other_a,blob)
    assert state_hash(decode(other_b,enh).state)==state_hash(expected)
    empty=synthetic_state(4,1).subset(np.empty(0,dtype=np.int64))
    c,data,expected=packet(tmp_path,4,0,empty)
    assert state_hash(decode(c,data).state)==state_hash(expected)
    decoder.close()


def rewrite(blob, mutate):
    length=struct.unpack("<Q",blob[8:16])[0]
    header=json.loads(blob[16:16+length]); mutate(header)
    encoded=canonical_json(header)
    result=blob[:8]+struct.pack("<Q",len(encoded))+encoded+blob[16+length:-32]
    return result+hashlib.sha256(result).digest()


def test_cache_hits_do_not_bypass_checksums_or_parent_ownership(tmp_path):
    d,blob,_=packet(tmp_path,1,0,synthetic_state(1,8))
    decoder=ProgressiveDecoder(optimization={"native":"memory"})
    state=decoder.decode_payload(d,blob).state
    bad=bytearray(blob);bad[-45]^=1
    with pytest.raises(ValueError): decoder.decode_payload(d,bad)
    decoder.release_frame("actor",1)
    # Existing stream cache must not override a changed expected raw digest.
    corrupt=rewrite(blob,lambda h:h["arrays"]["sh"].update(sha256="0"*64))
    with pytest.raises(ValueError,match="checksum"): decoder.decode_payload(d,corrupt)
    corrupt=rewrite(blob,lambda h:h.update(raw_sha256="0"*64))
    with pytest.raises(ValueError,match="checksum"): decoder.decode_payload(d,corrupt)
    assert not decoder.decoded_states
    state.arrays["xyz"].shape=(24,)
    with pytest.raises(ValueError,match="layout"): state.validate()


def test_memory_native_rejects_bad_stream_and_dimensions():
    from collections import Counter
    with pytest.raises(ValueError,match="rejected"):
        MemoryDraco().decode(b"not draco",2,12,Counter())


def test_read_only_stride_mutation_invalidates_verified_digest(tmp_path):
    d,blob,_=packet(tmp_path,1,0,synthetic_state(1,8))
    with ProgressiveDecoder(optimization={"native":"memory"}) as decoder:
        state=decoder.decode_payload(d,blob).state
        state.arrays["xyz"].strides=(0,0)
        with pytest.raises(ValueError,match="layout"):
            _=state.verified_digest


def test_cache_enforces_byte_and_entry_limits():
    cache=ByteCache(8)
    cache.put("a",b"123456",6); cache.put("b",b"abc",3)
    assert cache.get("a") is None and cache.size==3
    cache.put("large",b"x"*9,9)
    assert cache.get("large") is None
    for i in range(5000):cache.put(i,b"",0)
    assert len(cache.items)<=4096


def test_stream_package_reassembles_exact_packets_and_supports_seeks(tmp_path):
    from tools.content_preparation.packaging import package_object
    from tools.build_static_dynamic import build
    from src.client.static_dynamic import StaticDynamicStore
    root=tmp_path/"actor";root.mkdir()
    codec=get_codec({"name":"gaussian_attribute_draco_byteplanes"})
    frames=[{"frame":f,"timestamp":f/30} for f in (1,2,3)]
    records=[]; originals={}
    for f in (1,2,3):
        parent=None
        for q in (0,1):
            state=synthetic_state(1,4+q*3)
            state.arrays["xyz"]+=np.float32(f)
            if f==3:state.arrays["opacity"]+=np.float32(.25)
            state.metadata.update(frame=f,quality=f"Q{q}")
            path=root/f"{f}-{q}.cpgs"
            record=codec.encode(state,path,parent=parent)
            parent=codec.decode(path,parent)
            identifier=f"progressive:{'Base' if q==0 else 'E1'}:{f}"
            records.append({**record,"id":identifier,"mode":"progressive","frame":f,
                            "quality":f"Q{q}","path":path.name})
            originals[identifier]=path.read_bytes()
    package=package_object("actor",frames,["Q0","Q1"],records,{"gof_frames":3,"segment_frames":2},root)
    (root/"package_index.json").write_text(json.dumps(package))
    result=build(root/"package_index.json",tmp_path/"variant")
    store=StaticDynamicStore(result["index"])
    for f in (3,1,2,3):
        for record in package["payloads"]:
            if record["frame"]==f:
                data=store.read("actor:"+record["id"],record["sha256"],record["bytes"])
                assert data==originals[record["id"]]
    assert store.stats["cache_hits"]>0
    assert "sh" in store.document["invariance"]["actor:0:Q0"]["static_streams"]
    assert "opacity" not in store.document["invariance"]["actor:0:Q0"]["static_streams"]
    record=package["payloads"][0]
    import asyncio
    from types import SimpleNamespace
    class Client:
        calls=0
        async def fetch(self,spec):
            self.calls+=1
            return SimpleNamespace(success=True,error=None,body=(store.root/"streams"/spec.url.rsplit("/",1)[-1]).read_bytes())
    client=Client();store.cache.clear()
    async def fetch_twice():
        for _ in range(2):
            data=await store.read_http("actor:"+record["id"],record["sha256"],record["bytes"],client,"http://localhost/package")
            assert data==originals[record["id"]]
    asyncio.run(fetch_twice())
    assert client.calls==len({r["sha256"] for r in store.document["payloads"]["actor:"+record["id"]]["streams"]})
    with pytest.raises(ValueError,match="identity"):
        store.read("actor:"+record["id"],"0"*64,record["bytes"])
    reference=store.document["payloads"]["actor:"+record["id"]]["streams"][0]
    path=store.root/"streams"/reference["sha256"]
    content=path.read_bytes();path.write_bytes(bytes([content[0]^1])+content[1:]);store.cache.clear()
    with pytest.raises(ValueError,match="checksum"):
        store.read("actor:"+record["id"],record["sha256"],record["bytes"])


def test_bounded_workers_preserve_order_and_object_isolation(tmp_path):
    from src.client.decode_workers import DecodeWorkers
    chains=[]
    for f in (1,2,3):
        d,blob,reference=packet(tmp_path,f,0,synthetic_state(f,4))
        chains.append([(d,blob)])
    with DecodeWorkers(2,pending=2) as workers:
        results=list(workers.map(chains))
        assert [r.frame_index for r in results]==[1,2,3]
        assert len(workers.decoders)<=2
        for frame in results:assert state_hash(frame.state)==frame.descriptor.decoded_state_hash
    assert not workers.decoders


@pytest.mark.parametrize("native",["subprocess","memory"])
@pytest.mark.parametrize("verification",["strict","runtime"])
def test_float_bits_and_nonconsecutive_large_ids_are_preserved(tmp_path,native,verification):
    target=synthetic_state(1,16)
    values=np.array([0.,-0.,np.finfo(np.float32).smallest_subnormal,
                     -np.finfo(np.float32).smallest_subnormal,np.finfo(np.float32).max,
                     -np.finfo(np.float32).max,1.,-1.],dtype="<f4")
    for a in target.arrays.values():a.flat[:8]=values
    target.arrays["sh"]=np.resize(values,(16,3,16)).astype("<f4")
    target.ids=np.array([2**60+i for i in reversed(range(16))],dtype="<i8")
    d,blob,reference=packet(tmp_path,1,0,target)
    with ProgressiveDecoder(optimization={"native":native,"verification":verification}) as decoder:
        state=decoder.decode_payload(d,blob,expected_checksum=hashlib.sha256(blob).hexdigest()).state
        assert state_hash(state)==state_hash(reference)
        for name in target.arrays:assert state.arrays[name].tobytes()==target.arrays[name].tobytes()
        assert state.ids.tobytes()==target.ids.tobytes()


def test_runtime_requires_trusted_ingestion_and_remains_bit_exact(tmp_path):
    decoder = ProgressiveDecoder(optimization={"native":"memory", "verification":"runtime"})
    for frame in (1,2,3):
        parent = None
        for q,n in enumerate((4,7,10,13)):
            target = synthetic_state(1,n)
            target.arrays["xyz"] += np.float32(frame)
            d,blob,reference = packet(tmp_path,frame,q,target,parent,None if q==0 else f"actor:{frame}:{q-1}")
            if frame == 1 and q == 0:
                with pytest.raises(ValueError,match="trusted MPD"):
                    decoder.decode_payload(d,blob)
            actual = decoder.decode_payload(d,blob,expected_checksum=hashlib.sha256(blob).hexdigest()).state
            assert not actual.canonical_digest_computed
            with pytest.raises(ValueError,match="declared identity"):
                _ = actual.verified_digest
            assert state_hash(actual) == state_hash(reference)
            for name in actual.arrays:
                assert actual.arrays[name].tobytes() == reference.arrays[name].tobytes()
            parent = reference
    assert decoder.optimized.stats["array_hits"] > 0
    bad = bytearray(blob); bad[-45] ^= 1
    with pytest.raises(ValueError,match="trusted MPD"):
        decoder.decode_payload(d,bad,expected_checksum=hashlib.sha256(blob).hexdigest())
    decoder.close()


def test_runtime_corrupt_raw_stream_and_bounds_are_rejected(tmp_path):
    d,blob,_ = packet(tmp_path,1,0,synthetic_state(1,8))
    for mutate in (lambda h:h["arrays"]["sh"].update(sha256="0"*64),
                   lambda h:h["compressor"]["streams"]["sh"].update(offset=2**60)):
        corrupt = rewrite(blob,mutate)
        decoder = ProgressiveDecoder(optimization={"native":"memory","verification":"runtime"})
        with pytest.raises(ValueError):
            decoder.decode_payload(d,corrupt,expected_checksum=hashlib.sha256(corrupt).hexdigest())
        assert not decoder.decoded_states
        decoder.close()


def test_shared_runtime_workers_reuse_only_immutable_content_and_reject_layout_mutation(tmp_path):
    from src.client.decode_workers import DecodeWorkers
    chains = []
    for f in (1,2,3):
        d,blob,_ = packet(tmp_path,f,0,synthetic_state(1,8))
        chains.append([(d,blob,hashlib.sha256(blob).hexdigest())])
    with DecodeWorkers(2,optimization={"native":"memory","verification":"runtime"}) as pool:
        frames = list(pool.map(chains))
        for frame in frames:
            assert state_hash(frame.state)==frame.descriptor.decoded_state_hash
        assert len({id(decoder.optimized.streams) for decoder in pool.decoders})==1
        frames[-1].state.arrays["sh"].shape = (24,1)
        d,blob,_ = packet(tmp_path,4,0,synthetic_state(1,8))
        with pytest.raises(ValueError,match="layout"):
            list(pool.map([[(d,blob,hashlib.sha256(blob).hexdigest())]]))


def test_worker_selection_respects_memory_hardware_and_profile(tmp_path,monkeypatch):
    import src.client.decode_workers as module
    monkeypatch.setattr(module,"physical_cores",lambda:6)
    profile = tmp_path/"profile.json"
    profile.write_text(json.dumps({"physical_cores":6,"fps":{"runtime":{"3":{"1":8,"2":12,"4":16,"6":15}}}}))
    assert module.select_workers("auto",3,memory_mib=4096,profile=profile,verification="runtime")[0]==4
    assert module.select_workers("auto",3,memory_mib=800)[0]==1
    with pytest.raises(ValueError,match="cap"):
        module.select_workers(6,3,memory_mib=800)


def test_lookahead_capacity_is_configurable_and_memory_bounded():
    from src.client.decode_workers import queue_capacity
    assert queue_capacity(3, 4, 4096, 256, 37, 37) == 37
    assert queue_capacity(3, 4, 4096, 256, 10000) <= 42
    with pytest.raises(ValueError, match='memory cap'):
        queue_capacity(3, 4, 2048, 256, 100, 100)


def _wait_for_shared_cache_waiters(cache, key, count):
    import time
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        with cache.lock:
            flight = cache._flights.get(key)
            if flight is not None and flight["waiters"] == count:
                return
        time.sleep(.001)
    raise AssertionError("Concurrent cache requests did not attach to producer")


def test_shared_stream_singleflight_publishes_only_once_and_does_not_block_other_keys():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    cache = ByteCache(16)
    entered, release = threading.Event(), threading.Event()
    produced = []
    def compute():
        produced.append(True)
        entered.set()
        assert release.wait(5)
        return b"verified"
    def publish(value): cache.put("same",value,len(value))
    with ThreadPoolExecutor(max_workers=4) as executor:
        owner = executor.submit(cache.get_or_compute,"same",compute,publish)
        assert entered.wait(5)
        waiters = [executor.submit(cache.get_or_compute,"same",compute,publish) for _ in range(2)]
        try:
            _wait_for_shared_cache_waiters(cache,"same",2)
            other = executor.submit(cache.get_or_compute,"other",lambda:b"ok",lambda value:cache.put("other",value,len(value)))
            assert other.result(timeout=5) == (b"ok","miss")
        finally:
            release.set()
        assert owner.result(timeout=5) == (b"verified","miss")
        assert all(future.result(timeout=5) == (b"verified","shared") for future in waiters)
    assert len(produced) == 1
    assert cache.get_or_compute("same",compute,publish) == (b"verified","hit")
    assert cache.size <= cache.budget and not cache._flights


def test_shared_stream_singleflight_failed_producer_releases_waiters_and_allows_retry():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    cache = ByteCache(16)
    entered, release = threading.Event(), threading.Event()
    def invalid():
        entered.set()
        assert release.wait(5)
        raise ValueError("raw checksum mismatch")
    def publish(value): cache.put("same",value,len(value))
    with ThreadPoolExecutor(max_workers=3) as executor:
        owner = executor.submit(cache.get_or_compute,"same",invalid,publish)
        assert entered.wait(5)
        waiters = [executor.submit(cache.get_or_compute,"same",invalid,publish) for _ in range(2)]
        try:
            _wait_for_shared_cache_waiters(cache,"same",2)
        finally:
            release.set()
        for future in (owner,*waiters):
            with pytest.raises(ValueError,match="checksum"):
                future.result(timeout=5)
    assert cache.get("same") is None and not cache._flights
    assert cache.get_or_compute("same",lambda:b"valid",publish) == (b"valid","miss")


def test_shared_stream_singleflight_clear_and_inflight_limit_are_bounded():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    cache = ByteCache(8)
    cache.max_flights = 1
    entered, release = threading.Event(), threading.Event()
    fallback_entered = threading.Event()
    def blocked():
        entered.set()
        assert release.wait(5)
        return b"old"
    def fallback():
        fallback_entered.set()
        assert release.wait(5)
        return b"new"
    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(cache.get_or_compute,"old",blocked,lambda value:cache.put("old",value,len(value)))
        assert entered.wait(5)
        try:
            # A full flight table falls back to independent work without waiting.
            independent = executor.submit(cache.get_or_compute,"new",fallback,lambda value:cache.put("new",value,len(value)))
            assert fallback_entered.wait(5)
            assert len(cache._flights) == 1
            cache.clear()
        finally:
            release.set()
        assert owner.result(timeout=5) == (b"old","miss")
        assert independent.result(timeout=5) == (b"new","miss")
    assert cache.size == 0 and not cache.items and not cache._flights


@pytest.mark.parametrize("verification",["strict","runtime"])
def test_singleflight_native_streams_remain_exact_with_independent_object_states(tmp_path,verification,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    from src.client.optimized_codec import SharedCodecCache
    import threading
    d,blob,reference = packet(tmp_path,1,0,synthetic_state(1,128))
    shared = SharedCodecCache()
    decoders = [ProgressiveDecoder(optimization={"native":"memory","verification":verification,"shared_cache":shared}) for _ in range(4)]
    began,release,gate = threading.Event(),threading.Event(),threading.Lock()
    original = MemoryDraco.decode
    first = True
    def guarded(self,data,rows,width,stats):
        nonlocal first
        with gate:
            block = first
            first = False
        if block:
            began.set()
            assert release.wait(5)
        return original(self,data,rows,width,stats)
    monkeypatch.setattr(MemoryDraco,"decode",guarded)
    def decode(index):
        descriptor = replace(d,payload_id=f"object-{index}:1:0",object_id=f"object-{index}")
        return decoders[index].decode_payload(descriptor,blob,expected_checksum=hashlib.sha256(blob).hexdigest())
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(decode,i) for i in range(4)]
            assert began.wait(5)
            try:
                # Each object reaches the same first stream after its own input checks.
                with shared.streams.lock:
                    key = next(iter(shared.streams._flights))
                _wait_for_shared_cache_waiters(shared.streams,key,3)
            finally:
                release.set()
            frames = [future.result(timeout=5) for future in futures]
        totals = sum((Counter(decoder.optimized.stats) for decoder in decoders),Counter())
        header = decoders[0].optimized.inspect(blob,externally_verified=True)[0]
        unique_nonempty = {(r["sha256"],header["arrays"][name]["sha256"]) for name,r in header["compressor"]["streams"].items() if header["arrays"][name]["bytes"]}
        assert totals["native_invocations"] == len(unique_nonempty)
        assert totals["stream_singleflight_reuses"] >= 3
        for index,frame in enumerate(frames):
            assert frame.object_id == f"object-{index}"
            assert state_hash(frame.state) == state_hash(reference)
            for name in reference.arrays:
                assert frame.state.arrays[name].tobytes() == reference.arrays[name].tobytes()
        assert not shared.streams._flights
    finally:
        release.set()
        for decoder in decoders: decoder.close()
        shared.clear()


@pytest.mark.parametrize("reuse",[False,True])
def test_verified_index_classification_reuse_preserves_order_and_shape_checks(tmp_path,reuse):
    decoder = ProgressiveDecoder(optimization={"native":"memory","verification":"runtime","reuse_index_classification":reuse})
    try:
        for frame,count in enumerate((8,8,11,8),1):
            parent = None
            for quality in (0,1):
                state = synthetic_state(1,count+quality*3)
                state.arrays["xyz"] += np.float32(frame)
                if quality:
                    state = state.subset(np.arange(state.count)[::-1])
                d,blob,reference = packet(tmp_path,frame,quality,state,parent,None if not quality else f"actor:{frame}:0")
                actual = decoder.decode_payload(d,blob,expected_checksum=hashlib.sha256(blob).hexdigest()).state
                assert state_hash(actual) == state_hash(reference)
                parent = reference
        assert bool(decoder.optimized.stats["index_classification_hits"]) == reuse
    finally:
        decoder.close()


def test_finalization_benchmark_plan_distinguishes_archive_from_current_ablation(tmp_path):
    from types import SimpleNamespace
    from tools.benchmark_decoder_finalization import conditions
    args = SimpleNamespace(baseline_source=None,qualities=[0,1,2,3],worker_counts=[1,2,4,6],
                           no_serial=False,no_worker_ablation=False,condition=None)
    planned = conditions(args)
    assert len(planned) == 40 and len({name for name,_ in planned}) == 40
    assert planned[0][0] == "disabled-serial"
    assert "--decoder-source" not in planned[0][1]
    args.baseline_source = tmp_path/"archived.py"
    args.condition = ["before-q3-w4"]
    selected = conditions(args)
    assert len(selected) == 1
    name,flags = selected[0]
    assert name == "before-q3-w4"
    assert flags[flags.index("--decoder-source")+1] == str(args.baseline_source.resolve())
    args.condition = ["nonexistent"]
    with pytest.raises(ValueError,match="Unknown conditions"):
        conditions(args)


def test_archived_benchmark_loader_records_and_executes_actual_snapshot(tmp_path,monkeypatch):
    import sys
    import src.client.optimized_codec as codec
    from tools.benchmark_runtime_playback import decoder_implementation
    source = tmp_path/"archived.py"
    source.write_text('class OptimizedDraco:\n    def __init__(self, config=None, *, marker):\n        self.marker = marker\n')
    original = codec.OptimizedDraco
    monkeypatch.setattr(codec,"OptimizedDraco",original)
    monkeypatch.setitem(sys.modules,"decoder_benchmark_snapshot",None)
    metadata = decoder_implementation(source)
    instance = codec.OptimizedDraco(marker="snapshot executed",singleflight=False,reuse_index_classification=False)
    assert instance.marker == "snapshot executed"
    assert metadata["class"] == "decoder_benchmark_snapshot.OptimizedDraco"
    assert metadata["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert set(metadata["compatibility_options_ignored"]) == {"singleflight","reuse_index_classification"}
    assert codec.OptimizedDraco is not original
    # Only known compatibility options are omitted; unexpected options fail.
    with pytest.raises(TypeError):
        codec.OptimizedDraco(marker="unused",unexpected=True)
