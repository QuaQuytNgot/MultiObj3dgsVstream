"""Resume only complete Gaussian browser checks bound to exact viewer inputs."""
import copy
import unittest

from tools.validate_trial_viewer import reusable_gaussian_evidence


class GaussianViewerResumeTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {"variants": [{"objects": [{"id": "object", "representations": [
            {"quality": "Q0", "viewer_assets": [
                {"frame": 1, "decoded_state_hash": "state-0", "gaussian_count": 10},
                {"frame": 2, "decoded_state_hash": "state-1", "gaussian_count": 11},
            ]},
            {"quality": "Q1", "viewer_assets": [
                {"frame": 1, "decoded_state_hash": "state-2", "gaussian_count": 20},
                {"frame": 2, "decoded_state_hash": "state-3", "gaussian_count": 21},
            ]},
        ]}]}]}
        self.files = [{"path": "index.html", "bytes": 12, "sha256": "ui-hash"}]
        self.report = {"schema": "content-preparation.gaussian-viewer-validation.v1", "status": "failed",
                       "catalog_sha256": "catalog-hash", "viewer_files": copy.deepcopy(self.files),
                       "conditions_expected": 4, "conditions_checked": 4,
                       "checks": [
                           {"object": "object", "quality": quality, "frame": frame, "state_hash": state,
                            "gaussian_count": count, "nonblack_pixels": 100, "camera_preserved": True}
                           for quality, rows in (("Q0", [(1, "state-0", 10), (2, "state-1", 11)]),
                                                 ("Q1", [(1, "state-2", 20), (2, "state-3", 21)]))
                           for frame, state, count in rows],
                       "navigation": {"rotation": True, "pan": True, "zoom": True},
                       "webgl": {"version": "WebGL 2.0 test"}, "page_errors": [], "http_errors": [],
                       "rapid_switch": {"passed": True, "single_mesh": True, "final_quality": "Q1"}}

    def test_complete_matching_evidence_is_reusable(self):
        self.assertTrue(reusable_gaussian_evidence(self.report, "catalog-hash", self.files, self.catalog))

    def test_incomplete_or_stale_evidence_is_not_reusable(self):
        cases = []
        stale_catalog = copy.deepcopy(self.report)
        stale_catalog["catalog_sha256"] = "old-catalog"
        cases.append((stale_catalog, "catalog-hash", self.files, self.catalog))
        stale_viewer = copy.deepcopy(self.files)
        stale_viewer[0]["sha256"] = "changed-ui"
        cases.append((self.report, "catalog-hash", stale_viewer, self.catalog))
        missing_check = copy.deepcopy(self.report)
        missing_check["checks"].pop()
        cases.append((missing_check, "catalog-hash", self.files, self.catalog))
        black_asset = copy.deepcopy(self.report)
        black_asset["checks"][0]["nonblack_pixels"] = 0
        cases.append((black_asset, "catalog-hash", self.files, self.catalog))
        missing_navigation = copy.deepcopy(self.report)
        missing_navigation["navigation"]["pan"] = False
        cases.append((missing_navigation, "catalog-hash", self.files, self.catalog))
        js_error = copy.deepcopy(self.report)
        js_error["page_errors"] = ["uncaught error"]
        cases.append((js_error, "catalog-hash", self.files, self.catalog))
        for args in cases:
            with self.subTest(report=args[0]):
                self.assertFalse(reusable_gaussian_evidence(*args))


if __name__ == "__main__":
    unittest.main()
