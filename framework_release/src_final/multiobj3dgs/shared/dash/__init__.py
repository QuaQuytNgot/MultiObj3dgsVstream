"""Project DASH profile: CPSEG GOP bundles plus sidecar asset closures.

This is a project extension; a generic HEVC DASH player cannot decode it.
"""
from __future__ import annotations
from fractions import Fraction
import xml.etree.ElementTree as ET
from ..metadata.serialization import PROFILE, INDEX_SCHEME

NAMESPACE='urn:mpeg:dash:schema:mpd:2011'
ET.register_namespace('',NAMESPACE)


def _duration(seconds):
    return 'PT'+format(seconds,'.12f').rstrip('0').rstrip('.')+'S'


def _element(parent,name,**attributes):
    return ET.SubElement(parent,f'{{{NAMESPACE}}}{name}',{k:str(v) for k,v in attributes.items()})


def build_mpd(index, *, index_path='index.json', index_sha256):
    """Deterministic XML from the same normalized index consumed by the client."""
    root=ET.Element(f'{{{NAMESPACE}}}MPD',{'id':index['revision'],'type':'static','profiles':PROFILE,
        'minBufferTime':'PT1S','mediaPresentationDuration':_duration(index['duration'])})
    period=_element(root,'Period',id='scene',start='PT0S',duration=_duration(index['duration']))
    for adaptation_id,content in enumerate(index['contents'],1):
        rate=Fraction(str(content['fps'])).limit_denominator(1000000)
        adaptation=_element(period,'AdaptationSet',id=adaptation_id,mimeType='application/octet-stream',segmentAlignment='true')
        _element(adaptation,'EssentialProperty',schemeIdUri=PROFILE,value=content['id'])
        for representation in content['representations']:
            rep=_element(adaptation,'Representation',id=content['id']+':'+representation['id'],bandwidth=representation['bandwidth'],codecs='streamstgs')
            _element(rep,'SupplementalProperty',schemeIdUri='urn:multiobj3dgs:crf:1',value=representation['crf'])
            segments=_element(rep,'SegmentList',timescale=rate.numerator)
            timeline=_element(segments,'SegmentTimeline')
            for segment in representation['segments']:
                start=round(segment['start']*rate.numerator); duration=round(segment['duration']*rate.numerator)
                if abs(start-segment['start']*rate.numerator)>1e-6 or abs(duration-segment['duration']*rate.numerator)>1e-6:
                    raise ValueError('GOP time not representable in fps timescale')
                _element(timeline,'S',t=start,d=duration)
            for segment in representation['segments']:
                _element(segments,'SegmentURL',media=segment['resource']['url'])
    _element(root,'SupplementalProperty',schemeIdUri=INDEX_SCHEME,value=index_path+'#sha256='+index_sha256)
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)


def xml_shape(element):
    if element.text and element.text.strip(): raise ValueError('Unexpected text in MPD')
    if element.tail and element.tail.strip(): raise ValueError('Unexpected tail in MPD')
    return (element.tag,tuple(sorted(element.attrib.items())),tuple(xml_shape(c) for c in element))
