"""Host-only checks for benchmark evidence, independent of CUDA and the player."""
import copy
import unittest

from profile_native import summarize, validate_source_mappings


class ProfileEvidenceTests(unittest.TestCase):
    def setUp(self):
        segments = [dict(id=f'gop-{g}', start=g, duration=1., frames=[
            dict(seconds=g+i/2, source_frame=100+g*2+i, model_time=i/4)
            for i in range(2)]) for g in range(2)]
        self.index = dict(duration=3.,contents=[dict(id='content',fps=2., duration=2., representations=[
            dict(id=rep, segments=copy.deepcopy(segments)) for rep in ('high', 'low')])],
            instances=[dict(id='a', content_id='content', loop=True),
                       dict(id='b', content_id='content', scene_start=.5,
                            media_offset=1., active_duration=2., loop=True)])
        # b starts in GOP 1, wraps to GOP 0 at scene 1.5, then becomes inactive.
        self.frames = [
            dict(frame_id=0, scene_time=0., source_frames=[self.source('a', 0, 100, 0., 0.)]),
            dict(frame_id=1, scene_time=.5, source_frames=[self.source('a', 0, 101, .25, .5),
                                                         self.source('b', 1, 102, 0., 1.)]),
            dict(frame_id=3, scene_time=1.5, source_frames=[self.source('a', 1, 103, .25, 1.5, 'low'),
                                                          self.source('b', 0, 100, 0., 0., 'low')]),
            dict(frame_id=5, scene_time=2.5, source_frames=[self.source('a', 0, 101, .25, .5, 'low')]),
        ]
        self.controls=[dict(action='quality',after_frame=2,instance_id=identifier,representation_id='low')
                       for identifier in ('a','b')]

    def validate(self,frames=None):
        return validate_source_mappings(self.frames if frames is None else frames,self.index,
                                        fps=2.,quality_controls=self.controls)

    @staticmethod
    def source(instance, gop, frame, model, media, rep='high'):
        return dict(instance_id=instance, gop_id=f'gop-{gop}', representation_id=rep,
                    source_frame=frame, model_time=model, media_time=media)

    def test_independent_offsets_wrap_activation_and_representation(self):
        result = self.validate()
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['checked_objects'], 6)

    def test_wrong_source_frame_rejected(self):
        self.frames[2]['source_frames'][1]['source_frame'] += 1
        result = self.validate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['error_count'], 1)

    def test_missing_active_instance_rejected(self):
        self.frames[1]['source_frames'].pop()
        self.assertFalse(self.validate()['valid'])

    def test_legacy_unknown_and_partial_evidence_rejected(self):
        self.assertIsNone(self.validate([dict(frame_id=0)])['valid'])
        del self.frames[1]['source_frames']
        self.assertFalse(self.validate()['valid'])

    def test_common_completion_deadline_metric_ignores_player_flag(self):
        times = (10., 10.+1/30, 10.+2/30+.0005, 10.+3/30+.002)
        events = [dict(stage='runtime_configuration', presentation_fps=30.)]
        events += [dict(stage='scene_frame', fresh=True, frame_id=i, scene_time=i/30,
                        monotonic_s=value, duration_s=.01, deadline_missed=False)
                   for i, value in enumerate(times)]
        result = summarize({}, events, {}, [], 9.)
        self.assertEqual(result['frame_completion_miss_count_strict'], 2)
        self.assertEqual(result['frame_completion_miss_count_over_1ms'], 1)
        self.assertEqual(result['deadline_misses'], 0)

    def test_scene_loop_continuity_preserves_global_frame_ids(self):
        self.index['duration']=2.
        events=[dict(stage='runtime_configuration',presentation_fps=2.)]
        events += [dict(stage='scene_frame',fresh=True,frame_id=i,scene_time=(i%4)/2,
                        monotonic_s=10.+i/2,duration_s=.01) for i in range(8)]
        result=summarize({},events,{},[],9.,self.index,scene_loop=True)
        self.assertTrue(result['contiguous_frame_ids'])
        self.assertTrue(result['contiguous_timestamps'])
        self.assertFalse(summarize({},events,{},[],9.,self.index)['contiguous_timestamps'])
        events[5]['scene_time']+=.5
        self.assertFalse(summarize({},events,{},[],9.,self.index,scene_loop=True)['contiguous_timestamps'])


if __name__ == '__main__':
    unittest.main()
