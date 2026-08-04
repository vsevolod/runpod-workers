"""Unit tests for h3_infer.delivery."""

from __future__ import annotations

import unittest

from h3_infer.delivery import (
    BUCKET_ENV_KEYS,
    choose_delivery,
    bucket_configured,
)


class TestBucketConfigured(unittest.TestCase):
    def test_all_four(self):
        env = {k: "x" for k in BUCKET_ENV_KEYS}
        self.assertTrue(bucket_configured(env))

    def test_missing_name(self):
        env = {k: "x" for k in BUCKET_ENV_KEYS}
        del env["BUCKET_NAME"]
        self.assertFalse(bucket_configured(env))

    def test_only_endpoint(self):
        self.assertFalse(bucket_configured({"BUCKET_ENDPOINT_URL": "http://s3"}))


class TestChooseDelivery(unittest.TestCase):
    def test_url_when_bucket(self):
        self.assertEqual(choose_delivery(True, 99_000_000).mode, "url")

    def test_base64_small(self):
        self.assertEqual(choose_delivery(False, 7_000_000).mode, "base64")

    def test_error_large(self):
        plan = choose_delivery(False, 7_000_001)
        self.assertEqual(plan.mode, "error")
        self.assertIsNotNone(plan.error)


if __name__ == "__main__":
    unittest.main()
