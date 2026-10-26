"""Scientific telemetry: label reference-profile proxies without inventing PSNR."""


def viewport_quality_proxy(runtime, quality):
    """Use displayed representation identities and the last real visibility sample.

    This is not PSNR of a newly rendered interactive viewport: no corresponding
    ground-truth image exists. Keep that measurement null for downstream papers.
    """
    sources=runtime._last_frame_sources
    weights=runtime.visibility
    per_object={}
    for source in sources:
        identifier=source['instance_id']
        instance=next(i for i in runtime.catalog.instances if i.id==identifier)
        content=runtime.catalog.contents[instance.content_id]
        rep=content.representations[source['representation_id']]
        segment=next(s for s in rep.segments if s.id==source['gop_id'])
        per_object[identifier]={
            'content_id':content.id,'gop_index':segment.gop_index,'crf':rep.crf,
            'source_frame':source['source_frame'],
            'aggregated_reference_psnr_db':quality.psnr(content.id,segment.gop_index,rep.crf),
            'visibility_weight':weights.get(identifier)}
    total=sum(w for w in weights.values())
    proxy=sum((weights.get(i,0)/total)*row['aggregated_reference_psnr_db'] for i,row in per_object.items()) if total>0 else None
    return {'scene_time':runtime.displayed_time,'frame_id':runtime.frame_id-1,
            'metric':'visibility-weighted aggregated reference PSNR proxy',
            'weighted_reference_psnr_db':proxy,'actual_rendered_viewport_psnr_db':None,
            'camera_policy':'current_pose_hold',
            'quality_policy':'all-reference-view arithmetic mean in dB; no interactive-view correspondence',
            'per_object':per_object}
