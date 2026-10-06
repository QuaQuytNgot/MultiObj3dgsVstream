import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.evaluate_decoded_gt import read_gt, sample_from_pose
from tools.content_preparation.renderer_adapter import camera_parameters


class DecodedGTTests(unittest.TestCase):
    def test_matching_camera_and_roll_rejection(self):
        settings = {"up_axis":"z", "width":64, "height":64, "fov_degrees":39.6}
        expected = {"azimuth":120., "elevation":20., "distance":4.03, "scale":1.}
        cam = camera_parameters(expected, settings)
        pose = np.linalg.inv(cam.world_view).astype(np.float64)
        pose[:3,1:3] *= -1
        sample, error = sample_from_pose(pose,settings)
        self.assertLess(error,2e-5)
        np.testing.assert_allclose(camera_parameters(sample,settings).world_view,cam.world_view,atol=1e-6)
        roll = np.eye(4)
        roll[:2,:2] = [[np.cos(.2),-np.sin(.2)],[np.sin(.2),np.cos(.2)]]
        with self.assertRaisesRegex(ValueError,"incompatible"):
            sample_from_pose(pose@roll,settings)
        with self.assertRaisesRegex(ValueError,"Z-up"):
            sample_from_pose(pose,{**settings,"up_axis":"y"})

    def test_rgba_composite_without_silent_resolution_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"image.png"
            rgba=np.zeros((32,32,4),dtype=np.uint8)
            rgba[...,0]=255
            rgba[...,3]=128
            Image.fromarray(rgba).save(path)
            rgb=read_gt(path,[0,0,0],32,32)
            self.assertEqual(rgb.dtype,np.float32)
            self.assertAlmostEqual(float(rgb[0,0,0]),128/255,places=6)
            self.assertEqual(float(rgb[...,1:].max()),0)
            with self.assertRaisesRegex(ValueError,"no automatic resize"):
                read_gt(path,[0,0,0],64,64)


if __name__=="__main__":
    unittest.main()
