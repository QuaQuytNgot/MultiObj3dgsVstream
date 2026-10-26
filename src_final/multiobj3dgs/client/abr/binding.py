"""Bind HTTP-verified packages to the exact offline measured representations."""
import asyncio
import time
from .network import PipelineResourceCache


class QualityBoundCache(PipelineResourceCache):
    def bind_quality(self, catalog, quality):
        self.quality=quality
        self.identities={segment.id:(content.id,segment.gop_index,rep.crf)
            for content in catalog.contents.values() for rep in content.representations.values()
            for segment in rep.segments}

    async def materialize(self, segment):
        package=await super().materialize(segment)
        try:
            began=time.monotonic()
            await asyncio.to_thread(self.quality.verify_materialized,*self.identities[segment.id],package)
            if self.events:self.events.emit('measured_representation_binding',segment_id=segment.id,duration_s=time.monotonic()-began)
        except BaseException:
            self.release_package(package)
            raise
        return package
