"""Checks that must pass before a price-test effect is reported."""

from dataclasses import dataclass, field

import pandas as pd
from scipy import stats


@dataclass
class GuardrailReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def sample_ratio_mismatch(n_a: int, n_b: int, expected_share_b: float = 0.5) -> float:
    """Chi-square p-value that the observed arm split matches the designed split."""
    total = n_a + n_b
    expected = [total * (1 - expected_share_b), total * expected_share_b]
    return float(stats.chisquare([n_a, n_b], f_exp=expected).pvalue)


def check_guardrails(n_a: int, n_b: int, *, min_per_arm: int = 1000, expected_share_b: float = 0.5,
                     srm_alpha: float = 0.001, pre_period_ok: bool = True) -> GuardrailReport:
    """Run the readout guardrails.

    - minimum arm size, so the CI is not built on a handful of units;
    - no sample ratio mismatch (SRM), which signals broken randomisation or logging;
    - the pre-period covariate is measured strictly before the test started.
    """
    srm_p = sample_ratio_mismatch(n_a, n_b, expected_share_b)
    report = GuardrailReport(
        checks={
            "min_arm_size": min(n_a, n_b) >= min_per_arm,
            "no_srm": srm_p >= srm_alpha,
            "covariate_pre_period": bool(pre_period_ok),
        },
        details={"srm_p_value": srm_p, "n_a": float(n_a), "n_b": float(n_b)},
    )
    return report


def pre_period_only(timestamps: pd.Series, test_start) -> bool:
    """True if every covariate timestamp is strictly before the test start."""
    ts = pd.to_datetime(timestamps)
    return bool((ts < pd.Timestamp(test_start)).all())
