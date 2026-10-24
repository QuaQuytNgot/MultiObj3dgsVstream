"""Shared application-body bandwidth trace; not a kernel-level network emulator."""
import math
import threading
import time

class BodyPacer:
    def __init__(self,points,clock=time.monotonic,sleep=time.sleep):
        self.points=tuple((float(p['at_s']),float(p['bps']),float(p.get('latency_s',0))) for p in points)
        if not self.points or self.points[0][0]!=0 or any(not all(math.isfinite(x) for x in p) or p[0]<0 or p[1]<=0 or p[2]<0 for p in self.points) or any(a[0]>=b[0] for a,b in zip(self.points,self.points[1:])):
            raise ValueError('Trace needs increasing at_s from zero, positive bps and nonnegative latency_s')
        self.clock,self.sleep=clock,sleep;self.epoch=clock();self.next_at=0.;self.lock=threading.Lock()

    def latency(self):
        now=self.clock()-self.epoch
        point=next((p for p in reversed(self.points) if p[0]<=now),self.points[0])
        self.sleep(point[2])

    def reserve(self,size,now):
        with self.lock:
            position=max(now,self.next_at);bits=size*8
            while bits>0:
                index=max(i for i,p in enumerate(self.points) if p[0]<=position)
                rate=self.points[index][1]
                boundary=self.points[index+1][0] if index+1<len(self.points) else float('inf')
                available=(boundary-position)*rate
                if bits<=available:
                    position+=bits/rate;bits=0
                else:
                    bits-=available;position=boundary
            self.next_at=position
            return position

    def pace(self,size):
        target=self.reserve(size,self.clock()-self.epoch)
        self.sleep(max(0.,self.epoch+target-self.clock()))
