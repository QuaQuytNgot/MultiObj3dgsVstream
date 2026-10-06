"""Readiness must distinguish optional browser coverage from missing core checks."""
import unittest
from types import SimpleNamespace

from tools.content_preparation.self_test import successful


class SelfTestReadinessTests(unittest.TestCase):
    def test_optional_browser_skips_are_reported_but_not_required(self):
        result = SimpleNamespace(wasSuccessful=lambda: True, skipped=[
            (None, "Optional browser verification requires installed Playwright"),
            (None, "Sandbox forbids localhost sockets; run optional browser verification outside that sandbox"),
        ])
        self.assertTrue(successful(result))
        self.assertFalse(successful(result, strict_optional=True))

    def test_missing_core_or_lpips_coverage_is_required(self):
        result = SimpleNamespace(wasSuccessful=lambda: True, skipped=[
            (None, "Offline LPIPS integration requires cached official weights"),
        ])
        self.assertFalse(successful(result))

    def test_test_failures_still_fail_readiness(self):
        result = SimpleNamespace(wasSuccessful=lambda: False, skipped=[])
        self.assertFalse(successful(result))
