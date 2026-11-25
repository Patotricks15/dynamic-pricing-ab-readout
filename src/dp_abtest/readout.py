"""Effect estimation for a two-arm price test.

The estimand is ΔR = E[R | arm B] − E[R | arm A], where R is the revenue per unit
of randomisation (for example, price × quantity per customer over the test window).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Effect:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n_a: int
    n_b: int


def difference_in_means(y_a, y_b, alpha: float = 0.05) -> Effect:
    """Welch difference in means of arm B minus arm A, with a t-based CI."""
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    n_a, n_b = len(y_a), len(y_b)
    if n_a < 2 or n_b < 2:
        raise ValueError("each arm needs at least 2 observations")

    v_a = y_a.var(ddof=1) / n_a
    v_b = y_b.var(ddof=1) / n_b
    se = np.sqrt(v_a + v_b)
    diff = y_b.mean() - y_a.mean()

    # Welch-Satterthwaite degrees of freedom
    dof = (v_a + v_b) ** 2 / (v_a**2 / (n_a - 1) + v_b**2 / (n_b - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    if se == 0:
        p_value = 1.0 if diff == 0 else 0.0
    else:
        p_value = 2 * stats.t.sf(abs(diff / se), dof)
    return Effect(
        estimate=float(diff),
        se=float(se),
        ci_low=float(diff - t_crit * se),
        ci_high=float(diff + t_crit * se),
        p_value=float(p_value),
        n_a=n_a,
        n_b=n_b,
    )


def cuped_theta(y, x) -> float:
    """CUPED coefficient θ = Cov(Y, X) / Var(X), estimated on the pooled sample."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    var_x = x.var(ddof=1)
    if var_x == 0:
        raise ValueError("pre-period covariate has zero variance; CUPED is undefined")
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def cuped_adjust(y, x, theta: float | None = None) -> np.ndarray:
    """Return Y' = Y − θ(X − E[X]), with E[X] estimated by the sample mean of X."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if theta is None:
        theta = cuped_theta(y, x)
    return y - theta * (x - x.mean())


def readout(df: pd.DataFrame, arm: str, outcome: str, arm_a, arm_b, covariate: str | None = None,
            alpha: float = 0.05) -> dict:
    """Unadjusted and (optionally) CUPED-adjusted effect of arm B versus arm A.

    `covariate` must be a pre-experiment measure of the same unit. Its mean is
    removed over the pooled sample, so the adjusted effect stays unbiased under
    randomisation.
    """
    sub = df[df[arm].isin([arm_a, arm_b])]
    a_mask = sub[arm] == arm_a
    unadjusted = difference_in_means(sub.loc[a_mask, outcome], sub.loc[~a_mask, outcome], alpha)
    result = {"unadjusted": unadjusted}
    if covariate is not None:
        y_adj = cuped_adjust(sub[outcome], sub[covariate])
        adjusted = difference_in_means(y_adj[a_mask.to_numpy()], y_adj[~a_mask.to_numpy()], alpha)
        result["cuped"] = adjusted
        result["theta"] = cuped_theta(sub[outcome], sub[covariate])
        result["variance_ratio"] = (adjusted.se / unadjusted.se) ** 2 if unadjusted.se > 0 else np.nan
    return result
