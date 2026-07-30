import math

import pytest

from scripts.summarize_quality_audit import parse_bool, rate, wilson_interval


def test_parse_bool_accepts_explicit_ledger_values():
    assert parse_bool("yes", "label") is True
    assert parse_bool("FALSE", "label") is False
    with pytest.raises(ValueError):
        parse_bool("maybe", "label")


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(8, 10)
    assert lower < 0.8 < upper
    assert lower == pytest.approx(0.490162, abs=1e-6)
    assert upper == pytest.approx(0.943318, abs=1e-6)


def test_zero_denominator_rate_is_explicitly_missing():
    result = rate(0, 0)
    assert result["estimate"] is None
    assert result["wilson_95_ci"] is None
    assert not math.isnan(result["denominator"])
