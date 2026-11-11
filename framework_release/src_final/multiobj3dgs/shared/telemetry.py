"""Thread-safe, versioned stage events; times use the monotonic clock."""
from collections import deque
from contextlib import contextmanager
import json
import platform
import threading
import time
from uuid import uuid4


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, max=None)
    def percentile(q):
        p = (len(values)-1)*q
        i = int(p)
        return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(p-i)
    return dict(count=len(values),mean=sum(values)/len(values),p50=percentile(.5),p95=percentile(.95),max=values[-1])


class EventLog:
    def __init__(self, path=None, run_id=None, capacity=100000):
        self.run_id = run_id or uuid4().hex
        self.events = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._stream = None
        if path is not None:
            from pathlib import Path
            path = Path(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            self._stream = path.open('x',encoding='utf8')
        self.emit('environment', python=platform.python_version(), platform=platform.platform())

    def emit(self, stage, **fields):
        event = dict(fields, schema='multiobj3dgs.event.v1',run_id=self.run_id,
                     stage=stage,monotonic_s=time.monotonic())
        with self._lock:
            self.events.append(event)
            if self._stream:
                self._stream.write(json.dumps(event,allow_nan=False,default=str)+'\n')
                self._stream.flush()
        return event

    @contextmanager
    def timed(self, stage, **fields):
        started = time.monotonic()
        try:
            yield
        except BaseException as exc:
            self.emit(stage,duration_s=time.monotonic()-started,success=False,error=str(exc),**fields)
            raise
        else:
            self.emit(stage,duration_s=time.monotonic()-started,success=True,**fields)

    def summary(self):
        with self._lock:
            events = list(self.events)
        stages = sorted({e['stage'] for e in events})
        return dict(schema='multiobj3dgs.summary.v1',run_id=self.run_id,
                    retained_events=len(events), stages={s:distribution([e['duration_s'] for e in events if e['stage']==s and 'duration_s' in e]) for s in stages})

    def close(self):
        with self._lock:
            if self._stream:
                self._stream.close()
                self._stream = None
