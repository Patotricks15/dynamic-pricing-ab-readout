# Dynamic Pricing: Price A/B Readout (CUPED)

**Question.** Does a price change produce the expected revenue change in a randomised test?

**Status: blocked on data.** This repo contains the readout method, its guardrails, tests, and a notebook skeleton. No real randomised price test has been found. None of the three candidate datasets in the plan has arm assignment (see [data/README.md](data/README.md)). No effect has been estimated, and no result is reported.

This is project 3 of the dynamic pricing portfolio. It follows [dynamic-pricing-elasticity](https://github.com/Patotricks15/dynamic-pricing-elasticity) and [dynamic-pricing-optimal-price](https://github.com/Patotricks15/dynamic-pricing-optimal-price).

## Method

Estimand: `ΔR = E[R | arm B] − E[R | arm A]`, where `R` is revenue per unit of randomisation.

- **Unadjusted.** Welch difference in means, with a t-based 95% CI.
- **CUPED.** `Y' = Y − θ(X − E[X])`, with `θ = Cov(Y, X) / Var(X)` and `X` a pre-experiment covariate. It is used only when the covariate is measured before the test starts.
- **Guardrails.** A result is reported only if the arms reach a minimum size, there is no sample ratio mismatch (chi-square, α = 0.001), and the covariate is pre-period.
- **Cross-check.** OLS with HC3 errors and a cross-validated R² for the covariate (seed 42).

## Decision supported

Whether a planned price change is expected to raise revenue per unit, and whether the CI excludes zero. If the CI includes zero, the test does not support the change.

## Limitation

**No randomised price data is available here, so the readout has not been applied to any real test.** The notebook will run only once a real experiment log is attached in a Kaggle Notebook. Until then, these are code and tests only. The candidate datasets are observational or synthetic, so they cannot support a causal price claim.

## Reproduce

Nothing is downloaded. Data is read in place from `/kaggle/input/<slug>/` inside a Kaggle Notebook.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The notebook was **not executed locally**, because the data is not on this machine. `notebooks/ab_readout.py` is the source (jupytext percent format), and `notebooks/ab_readout.ipynb` is its unexecuted conversion. It fails with a clear error until the dataset slug is set and the file is attached.

## Layout

```
src/dp_abtest/readout.py      difference_in_means, cuped_theta, cuped_adjust, readout
src/dp_abtest/guardrails.py   sample_ratio_mismatch, check_guardrails, pre_period_only
tests/test_readout.py         hand-computed Welch CI, CUPED algebra, variance reduction, guardrails
notebooks/ab_readout.py       readout skeleton that reads /kaggle/input (jupytext percent format)
notebooks/ab_readout.ipynb    unexecuted notebook
reports/figures/              figures; empty until the notebook runs on real data
data/README.md                candidate review, expected input schema, publication rule
```
