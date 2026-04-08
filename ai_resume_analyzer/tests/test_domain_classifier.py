"""
tests/test_domain_classifier.py

Unit tests for the unified domain classifier module.
These tests mock the underlying backends to avoid requiring trained model files.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.domain_classifier import (
    DOMAIN_LABELS,
    get_available_domains,
    _DOMAIN_DESCRIPTIONS,
)


class TestDomainLabels:
    def test_domain_labels_is_list(self):
        assert isinstance(DOMAIN_LABELS, list)

    def test_expected_domains_present(self):
        expected = [
            "INFORMATION-TECHNOLOGY",
            "FINANCE",
            "HR",
            "HEALTHCARE",
            "ENGINEERING",
        ]
        for domain in expected:
            assert domain in DOMAIN_LABELS, f"Missing domain: {domain}"

    def test_get_available_domains_returns_copy(self):
        d1 = get_available_domains()
        d2 = get_available_domains()
        d1.append("TEST")
        assert "TEST" not in d2, "get_available_domains should return a copy"

    def test_all_domains_have_descriptions(self):
        for domain in DOMAIN_LABELS:
            assert domain in _DOMAIN_DESCRIPTIONS, (
                f"Domain '{domain}' has no description for zero-shot classifier"
            )

    def test_descriptions_are_nonempty(self):
        for domain, desc in _DOMAIN_DESCRIPTIONS.items():
            assert len(desc.strip()) > 5, f"Description for '{domain}' is too short"


class TestPredictDomainUnified:
    """Tests the routing logic using mocked backends."""

    def test_empty_text_raises(self):
        from inference.domain_classifier import predict_domain_unified
        with pytest.raises(ValueError):
            predict_domain_unified("")

    def test_whitespace_only_raises(self):
        from inference.domain_classifier import predict_domain_unified
        with pytest.raises(ValueError):
            predict_domain_unified("   ")

    def test_xgboost_route_by_default(self):
        """When USE_TRANSFORMER_CLASSIFIER=False, should call classify_domain_xgboost."""
        mock_result = {
            "predicted_domain": "FINANCE",
            "confidence": 0.85,
            "all_probabilities": {},
            "backend": "xgboost",
        }
        with patch("inference.domain_classifier.USE_TRANSFORMER_CLASSIFIER", False), \
             patch("inference.domain_classifier.classify_domain_xgboost",
                   return_value=mock_result) as mock_xgb:
            from inference.domain_classifier import predict_domain_unified
            result = predict_domain_unified("Financial analyst investment banking")

        assert result["backend"] == "xgboost"
        mock_xgb.assert_called_once()

    def test_transformer_fallback_on_failure(self):
        """If transformer raises, should fall back to XGBoost."""
        mock_xgb_result = {
            "predicted_domain": "ENGINEERING",
            "confidence": 0.78,
            "all_probabilities": {},
            "backend": "xgboost",
        }
        with patch("inference.domain_classifier.USE_TRANSFORMER_CLASSIFIER", True), \
             patch("inference.domain_classifier.classify_domain_transformer",
                   side_effect=RuntimeError("model not available")), \
             patch("inference.domain_classifier.classify_domain_xgboost",
                   return_value=mock_xgb_result):
            from inference.domain_classifier import predict_domain_unified
            result = predict_domain_unified("Mechanical engineer civil CAD design")

        assert result["backend"] == "xgboost"

    def test_result_has_required_keys(self):
        mock_result = {
            "predicted_domain": "HEALTHCARE",
            "confidence": 0.90,
            "all_probabilities": {"HEALTHCARE": 0.90},
            "backend": "xgboost",
        }
        with patch("inference.domain_classifier.USE_TRANSFORMER_CLASSIFIER", False), \
             patch("inference.domain_classifier.classify_domain_xgboost",
                   return_value=mock_result):
            from inference.domain_classifier import predict_domain_unified
            result = predict_domain_unified("Nurse doctor medical hospital clinical")

        assert "predicted_domain" in result
        assert "confidence" in result
        assert "backend" in result

    def test_transformer_used_when_flag_enabled(self):
        """When USE_TRANSFORMER_CLASSIFIER=True and transformer succeeds."""
        mock_result = {
            "predicted_domain": "INFORMATION-TECHNOLOGY",
            "confidence": 0.91,
            "all_probabilities": {},
            "backend": "transformers",
        }
        with patch("inference.domain_classifier.USE_TRANSFORMER_CLASSIFIER", True), \
             patch("inference.domain_classifier.classify_domain_transformer",
                   return_value=mock_result) as mock_tf:
            from inference.domain_classifier import predict_domain_unified
            result = predict_domain_unified("Python machine learning AWS")

        assert result["backend"] == "transformers"
        mock_tf.assert_called_once()
