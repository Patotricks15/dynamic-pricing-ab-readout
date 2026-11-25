# Data

Raw data is not committed. Only this README is tracked. Data is read in place from
`/kaggle/input/<slug>/` inside a Kaggle Notebook. Nothing is downloaded to the development machine.

## Candidate review (plan read 2026-11-17)

Candidates are the three named in the plan. None was downloaded for this repo. The plan records the
read date and the decisions below, not per-dataset licenses, except where noted.

| Kaggle slug | License | Declared origin | Randomised price test with arm assignment? | Decision |
|---|---|---|---|---|
| `thedevastator/online-retail-transaction-data` | "other" (per the sibling repo's data README; unconfirmed) | Invoice-line sales of a UK online gift retailer, 2010–2011 | No. Observational invoices with no arm column | Not used for DP03 |
| `bhanupratapbiswas/retail-price-optimization-case-study` | Not recorded in the plan; confirm on Kaggle before use | Case study: product-month prices and competitor prices | No. Product-month panel with no arm column | Not used for DP03 |
| `arashnic/dynamic-pricing-dataset` | Not recorded in the plan; confirm on Kaggle before use | Generated ride-hailing table | No. Rejected as synthetic | Rejected |

**Status: blocked.** No candidate has a randomised price test. The notebook is a skeleton, and no effect has been estimated.

## Expected input for the notebook

A real experiment log with one row per unit (customer, session, or product-store), with columns:

| Column | Meaning |
|---|---|
| `unit_id` | Unit of randomisation |
| `arm` | `A` (control price) or `B` (test price) |
| `assigned_at` | Assignment timestamp |
| `pre_revenue` | Revenue in the pre-period (CUPED covariate) |
| `pre_period_at` | Timestamp of the covariate observation; must be before the test start |
| `revenue` | Revenue per unit over the test window |

Attach the dataset in a Kaggle Notebook, then set `KAGGLE_SLUG` in `notebooks/ab_readout.py`.
Before use, record its slug, license, declared origin, and read date here.

## Publication rule

If a real randomised log is found with license "other" or a case-study origin, publish only code and figures,
state the license as unconfirmed, and link the source page.
