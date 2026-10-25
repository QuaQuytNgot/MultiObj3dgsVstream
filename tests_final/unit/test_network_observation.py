from dataclasses import replace
from multiobj3dgs.shared.transfer import RequestSpec,TransferResult
from multiobj3dgs.client.network.observation import TransferObserver
from multiobj3dgs.client.network.bandwidth import BandwidthEstimator

def result(name,start,end,size=100,kind='media',success=True):
    return TransferResult(RequestSpec(name,'https://test.invalid/x',metadata={'traffic_kind':kind}),200,b'x',None,size,start,start,end,0.,end-start,end-start,size*8/(end-start),size*8/(end-start),'HTTP/1.1',{},success,None,None,0.,'test')

def test_concurrent_goodput_uses_union_busy_time():
    observer=TransferObserver(concurrency=2)
    a,b=result('a',0.,2.),result('b',1.,3.)
    observer.started(a.spec);observer.started(b.spec)
    observer(a)
    assert observer.estimator.estimate_bandwidth().bps is None
    observer(b)
    assert observer.estimator.estimate_bandwidth().bps==1600/3
    assert observer.snapshot()['total_bytes']==200

def test_failed_partial_and_metadata_count_without_capacity_sample():
    observer=TransferObserver(concurrency=1)
    observer(result('metadata',0.,1.,kind='metadata'))
    observer(result('failed',0.,1.,size=12,success=False))
    assert observer.estimator.estimate_bandwidth().bps is None
    assert observer.snapshot()['failed_bytes']==12
    observer(result('media',0.,1.))
    assert observer.estimator.estimate_bandwidth().bps==800

def test_harmonic_mean_and_expiry():
    from multiobj3dgs.shared.transfer import to_bandwidth_sample
    now=[0.]
    estimator=BandwidthEstimator(clock=lambda:now[0],max_sample_age_s=1.)
    estimator.accept_transfer_sample(to_bandwidth_sample(result('a',0.,1.)))
    estimator.accept_transfer_sample(to_bandwidth_sample(result('b',0.,2.)))
    assert abs(estimator.estimate_bandwidth().bps-1600/3)<1e-6
    now[0]=2.
    assert estimator.estimate_bandwidth().stale
