"""Internal integrity handoff from verified immutable segment to member decoder.

The trusted MPD binds both the segment digest and member offsets/identities.
After one complete segment SHA-256, slicing its immutable bytes needs no second
bulk checksum scan in runtime mode. Strict mode still verifies each member.
"""
from dataclasses import dataclass
import hashlib
import re


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPayload:
    data: memoryview
    checksum: str
    request_unit_id: str

    def __init__(self, segment, location):
        if not isinstance(segment, VerifiedSegment) or location.request_unit_id != segment.unit_id:
            raise ValueError("Verified member belongs to another segment")
        if type(location.offset) is not int or type(location.length) is not int or location.offset < 0 or location.length < 0 or location.offset+location.length > len(segment.data):
            raise ValueError("Payload extent exceeds verified segment")
        if not isinstance(location.checksum,str) or re.fullmatch("[0-9a-f]{64}",location.checksum) is None:
            raise ValueError("Invalid expected payload checksum")
        object.__setattr__(self,"data",memoryview(segment.data)[location.offset:location.offset+location.length])
        object.__setattr__(self,"checksum",location.checksum)
        object.__setattr__(self,"request_unit_id",segment.unit_id)

    def __len__(self):
        return len(self.data)

    def detached(self):
        """Own only this member, so a queued frame cannot pin an evicted GB container."""
        token = object.__new__(type(self))
        object.__setattr__(token,"data",memoryview(bytes(self.data)))
        object.__setattr__(token,"checksum",self.checksum)
        object.__setattr__(token,"request_unit_id",self.request_unit_id)
        return token


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSegment:
    data: bytes
    checksum: str
    unit_id: str

    def __init__(self, unit, data):
        data = bytes(data)  # Detach mutable owners before hashing or publication.
        if len(data) != unit.size_bytes or hashlib.sha256(data).hexdigest() != unit.checksum:
            raise ValueError(f"Segment size/hash mismatch: {unit.unit_id}; file may still be transferring")
        object.__setattr__(self,"data",data)
        object.__setattr__(self,"checksum",unit.checksum)
        object.__setattr__(self,"unit_id",unit.unit_id)

    def member(self, location):
        return VerifiedPayload(self,location)

    def __len__(self):
        return len(self.data)
