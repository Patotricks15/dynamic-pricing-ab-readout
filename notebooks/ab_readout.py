# %% [markdown]
# # Price A/B readout (CUPED skeleton)
#
# **Question.** Does a price change produce the expected revenue change in a randomised test?
#
# **Status.** Skeleton only. None of the candidate datasets (see `data/README.md`) has a randomised
# price test with arm assignment, so no effect has been estimated. The notebook is written to run
# once a real experiment log is attached in a Kaggle Notebook under `/kaggle/input/`.
#
# **Estimand.** ΔR = E[R | arm B] − E[R | arm A], with a 95% CI. CUPED uses a pre-period revenue covariate.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from dp_abtest.guardrails import check_guardrails, pre_period_only
from dp_abtest.readout import readout

# Replace <slug> with the Kaggle dataset slug attached to the notebook.
KAGGLE_SLUG = "<slug>"
INPUT = Path(f"/kaggle/input/{KAGGLE_SLUG}/price_test.csv")
FIG = Path("/kaggle/working/figures")
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
ARM_A, ARM_B = "A", "B"
TEST_START = "2011-01-01"  # set to the first day of the randomised period

sns.set_theme(style="whitegrid")
np.random.seed(SEED)

if not INPUT.exists():
    raise FileNotFoundError(
        f"{INPUT} not found. Attach a randomised price-test dataset with slug {KAGGLE_SLUG!r} "
        "to this Kaggle Notebook (see data/README.md)."
    )

# %% [markdown]
# ## 1. Load the test table
#
# Expected columns: `unit_id`, `arm` (A or B), `assigned_at`, `pre_revenue` (pre-period covariate),
# `pre_period_at` (timestamp of the covariate observation, must be before the test start),
# `revenue` (outcome per unit over the test window).

# %%
df = pd.read_csv(INPUT, parse_dates=["assigned_at", "pre_period_at"])
df = df[df["arm"].isin([ARM_A, ARM_B])].dropna(subset=["revenue", "pre_revenue"]).copy()
counts = df["arm"].value_counts()
print(counts.to_string())

# %% [markdown]
# ## 2. Guardrails
#
# The effect is reported only if these pass: minimum arm size, no sample ratio mismatch,
# and the covariate is measured before the test started.

# %%
pre_ok = pre_period_only(df["pre_period_at"], TEST_START)
guard = check_guardrails(int(counts[ARM_A]), int(counts[ARM_B]), min_per_arm=1000,
                         expected_share_b=0.5, pre_period_ok=pre_ok)
print(guard.checks, guard.details)
assert guard.passed, "Guardrails failed; do not report an effect."

# %% [markdown]
# ## 3. Unadjusted and CUPED effect

# %%
res = readout(df, "arm", "revenue", ARM_A, ARM_B, covariate="pre_revenue")
rows = []
for label in ("unadjusted", "cuped"):
    e = res[label]
    rows.append({"estimator": label, "effect": e.estimate, "ci_low": e.ci_low,
                 "ci_high": e.ci_high, "p_value": e.p_value})
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print(f"theta = {res['theta']:.4f}; CUPED variance ratio = {res['variance_ratio']:.3f}")

# %% [markdown]
# ## 4. Cross-checks
#
# A regression-adjusted estimate with HC3 errors, and how well the covariate predicts the outcome
# out of fold (cross-validated R², seed 42).

# %%
ols = smf.ols("revenue ~ C(arm) + pre_revenue", data=df).fit(cov_type="HC3")
ci = ols.conf_int().loc[f"C(arm)[T.{ARM_B}]"]
print(f"OLS with covariate: effect {ols.params[f'C(arm)[T.{ARM_B}]']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

cv = cross_val_score(LinearRegression(), df[["pre_revenue"]], df["revenue"],
                     cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
print(f"covariate out-of-fold R² = {cv.mean():.3f}")

# %% [markdown]
# ## 5. Figure: arm-level revenue with CI

# %%
means = df.groupby("arm")["revenue"].agg(["mean", "sem"]).reindex([ARM_A, ARM_B])
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(means.index, means["mean"], yerr=1.96 * means["sem"], fmt="o", capsize=6)
ax.set_ylabel("Revenue per unit (mean, 95% CI)")
ax.set_title("Randomised price test: arm-level revenue (dataset: " + KAGGLE_SLUG + ")")
fig.tight_layout()
fig.savefig(FIG / "arm_revenue_ci.png", dpi=150)
plt.show()
