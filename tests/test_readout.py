import numpy as np
import pandas as pd
import pytest

from dp_abtest.guardrails import check_guardrails, pre_period_only, sample_ratio_mismatch
from dp_abtest.readout import cuped_adjust, cuped_theta, difference_in_means, readout


def test_difference_in_means_matches_hand_computed_welch_ci():
    # A=[1,2,3], B=[2,3,4]: diff=1, SE=sqrt(1/3+1/3), Welch df=4, t(0.975,4)=2.776445
    eff = difference_in_means([1, 2, 3], [2, 3, 4])
    half_width = 2.776445 * np.sqrt(2 / 3)
    assert eff.estimate == pytest.approx(1.0)
    assert eff.se == pytest.approx(np.sqrt(2 / 3))
    assert eff.ci_low == pytest.approx(1 - half_width, abs=1e-4)
    assert eff.ci_high == pytest.approx(1 + half_width, abs=1e-4)
    assert eff.p_value == pytest.approx(0.288, abs=1e-3)  # t=1.2247 with df=4


def test_difference_in_means_rejects_arm_with_one_unit():
    with pytest.raises(ValueError):
        difference_in_means([1.0], [2.0, 3.0])


def test_cuped_adjust_removes_covariate_part_with_given_theta():
    # Y = 2X + 1 for arm B. With theta=2 the adjusted outcome is constant within each arm.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 7.0, 9.0])
    adjusted = cuped_adjust(y, x, theta=2.0)
    assert adjusted.tolist() == pytest.approx([5.0, 5.0, 6.0, 6.0])


def test_cuped_theta_is_cov_over_var():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    assert cuped_theta(y, x) == pytest.approx(2.0)


def test_cuped_theta_rejects_constant_covariate():
    with pytest.raises(ValueError):
        cuped_theta([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])


def test_readout_cuped_reduces_variance_when_covariate_predicts_outcome():
    rng = np.random.default_rng(42)
    n = 200
    arm = np.repeat(["A", "B"], n // 2)
    pre = rng.normal(10, 2, n)
    effect = np.where(arm == "B", 0.5, 0.0)
    outcome = pre + effect + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"arm": arm, "pre_revenue": pre, "revenue": outcome})

    res = readout(df, "arm", "revenue", "A", "B", covariate="pre_revenue")
    assert res["unadjusted"].estimate == pytest.approx(0.5, abs=0.3)
    assert res["cuped"].estimate == pytest.approx(0.5, abs=0.2)
    assert res["variance_ratio"] < 0.1
    assert res["cuped"].ci_low < 0.5 < res["cuped"].ci_high


def test_sample_ratio_mismatch_flags_skewed_split():
    assert sample_ratio_mismatch(500, 500) == pytest.approx(1.0)
    assert sample_ratio_mismatch(6000, 4000) < 0.001


def test_guardrails_fail_on_small_arm_and_on_late_covariate():
    report = check_guardrails(50, 48, min_per_arm=1000, pre_period_ok=False)
    assert report.checks["min_arm_size"] is False
    assert report.checks["covariate_pre_period"] is False
    assert report.passed is False


def test_guardrails_pass_on_balanced_large_pre_period_test():
    report = check_guardrails(5000, 5000, min_per_arm=1000, pre_period_ok=True)
    assert report.passed is True


def test_pre_period_only_requires_strictly_before_start():
    ts = pd.Series(pd.to_datetime(["2011-01-01", "2011-01-31"]))
    assert pre_period_only(ts, "2011-02-01") is True
    assert pre_period_only(ts, "2011-01-31") is False
