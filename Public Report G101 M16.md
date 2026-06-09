# Machine Learning for Dynamic Bitcoin Allocation: A Risk-Adjusted, Long-Only Strategy

**Public Report — Group 101 (G101) · Module 16**
**Inteli — Instituto de Tecnologia e Liderança · Class T22 · 2025-2A**
**Business partner:** Hashdex · **Authors:** Group 101

> **Confidentiality note.** This is a public document. Confidential partner
> information — assets under management, client data, and proprietary
> implementation details (exact feature set and hyperparameters) — has been
> intentionally omitted. Performance figures reported here are **out-of-sample
> backtest and early paper-trading results**, not realized returns on capital.

---

## Abstract

We present a machine-learning system that decides, on a weekly basis, how much of a
portfolio to allocate to Bitcoin (BTC) versus a cash-equivalent benchmark (the
Brazilian interbank rate, CDI). The system is **long-only** (allocations between 0%
and 100%, no short selling) and is explicitly optimized for **downside
risk-adjusted return** rather than raw return. Its core is a dual ensemble of
gradient-boosted decision trees that forecasts short-horizon Bitcoin returns; the
forecast is translated into a position size that is scaled by a market-regime filter
and by the model's own confidence, and then bounded by automated risk controls. In a
4.3-year out-of-sample backtest, the strategy achieved substantially higher
downside-adjusted returns than static allocations while cutting the maximum drawdown
to roughly one-tenth of a buy-and-hold position. A battery of overfitting tests —
including walk-forward validation with temporal purging, deflated performance ratios,
and label-permutation tests — supports the conclusion that the signal is statistically
real. The system is currently in a paper-trading validation phase.

---

## 1. Introduction and problem

Bitcoin offers a long-run risk premium but with extreme volatility: peak-to-trough
losses of 65–80% are common. For institutional vehicles and most investors, that
drawdown profile is hard to hold. In practice, "how much Bitcoin to carry" is often a
**discretionary** decision — difficult to audit, to reproduce, and to scale.

This project addresses a focused question: **can the Bitcoin risk premium be captured
with a fraction of its drawdown, through a systematic, auditable process rather than
discretionary judgment?** The objective is not to maximize return at any cost, but to
maximize *risk-adjusted* performance under a long-only mandate.

---

## 2. Objectives

1. Build a systematic weekly BTC/CDI allocation signal that **dominates static
   allocations** on a downside-risk-adjusted basis.
2. Keep the maximum drawdown materially below a buy-and-hold position.
3. Ensure the process is **reproducible, validated against overfitting, and
   auditable** — suitable for review by a risk committee.
4. Operationalize the signal as a one-command production pipeline, and validate it in
   paper trading before any capital is committed.

Success is measured with portfolio-level, risk-adjusted metrics (Sortino ratio,
maximum drawdown, excess return versus benchmarks), not with classifier accuracy in
isolation — by design, the model's edge comes from *sizing* positions well, not from
predicting direction frequently.

---

## 3. Methodology

### 3.1 Data

The pipeline ingests data from a dozen public and licensed sources spanning three
families: **macro** (e.g., monetary aggregates and policy-rate proxies), **on-chain**
(network and valuation metrics), and **market/technical** (price, volume, derivatives
basis and funding). From these, the system derives roughly thirty engineered features,
including trend- and mean-reversion indicators and structural-regime measures. Feature
construction is strictly **backward-looking** — a property enforced by an automated
test that perturbs the most recent observation and verifies that no past feature value
changes, guarding against look-ahead bias.

### 3.2 Forecasting model

The forecasting core is a **dual ensemble** of gradient-boosted decision trees: one
ensemble regresses the short-horizon (multi-day) forward return of Bitcoin, while a
second classifies the direction of that move. Bagging many trees per ensemble reduces
seed-to-seed variance and makes the signal more stable in live use. The model family
was selected after comparing it against alternatives — including hidden Markov regime
models, recurrent neural networks, random forests, and stacked meta-learners — all of
which underperformed the chosen approach on the target objective.

### 3.3 Position sizing

The regression forecast is converted into a target allocation that is **scaled by two
factors**: (i) a *market-regime filter*, derived from long- and short-term moving
averages, which reduces exposure in bearish regimes; and (ii) a *confidence weight*
derived from the directional classifier, so the strategy bets larger only when the
model is more certain. The result is clipped to the [0%, 100%] long-only range. The
portfolio rebalances weekly, with an additional emergency rebalance triggered by very
large single-day moves. The model is retrained on a fixed semi-annual schedule using an
expanding window.

### 3.4 Risk controls

Three automated controls are applied to every signal: a **kill switch** that caps
exposure when cumulative drawdown breaches a threshold; an **accuracy de-risk** that
halves exposure when rolling predictive accuracy deteriorates while the model remains
overconfident; and a **population-stability monitor** that flags distribution drift in
the input features. These controls are part of the production signal, not an
afterthought.

### 3.5 Validation methodology

Validation follows established financial-ML practice:

- **Walk-forward, out-of-sample evaluation** with an expanding window and a temporal
  **purge/embargo** between train and test to prevent leakage.
- **Multi-seed** runs to quantify the variance of every reported metric.
- **Label-permutation testing**: the model is re-evaluated on randomly shuffled targets
  to establish a null distribution.
- **Deflated performance ratios** to discount the effect of having tried many
  configurations (multiple-testing).

---

## 4. Results

All figures below are **out-of-sample backtest** results over roughly 4.3 years
(248 weekly rebalances), reported on a multi-seed basis and net-aware of realistic
transaction costs (a few basis points per rebalance; results remain strong even under
pessimistic cost assumptions).

| Strategy | Sortino (daily) | Max drawdown | Outcome |
|---|---|---|---|
| Cash benchmark (CDI) | — | 0% | Low return, no risk |
| Static 30% BTC / 70% CDI | ~1.1 | ~-22% | Baseline blend |
| 100% Bitcoin (buy-and-hold) | ~0.6 | ~-67% | High risk |
| **This strategy (long-only ML)** | **~3.5** | **~-7%** | **Best risk-adjusted** |

The strategy delivered markedly higher downside-risk-adjusted performance than either
static allocation, while holding maximum drawdown to roughly **one-tenth** of a
buy-and-hold position — a direct result of spending a majority of the time in the cash
benchmark and concentrating exposure in favorable regimes.

**Statistical significance of the edge.** In label-permutation testing, **none of 100
shuffled-target runs beat the baseline** (p < 0.01), indicating the signal is not an
artifact of chance. After applying a deflated performance ratio to account for
multiple-testing, the strategy still passes under a realistic estimate of the number of
effective trials.

**Early live (paper) evidence.** In a strict out-of-sample window of the most recent
year-to-date (about 105 days), the strategy returned **+19.7%** while Bitcoin
buy-and-hold returned **-14.4%** and the cash benchmark **+4.1%** — early corroboration,
though concentrated in a small number of rebalances.

---

## 5. Discussion and limitations

We are deliberately conservative about these results:

- **Deflation.** Backtest metrics overstate live expectations. After discounting for
  multiple-testing, the *realistic* forward expectation is a Sortino in the 1.5–2.5
  range and a drawdown of -15% to -25% — still superior to the static baselines, but
  well below the raw backtest.
- **Concentration.** A small number of weeks account for a large share of the cumulative
  return; missing a few of them would materially reduce the edge.
- **Regime risk.** The model depends on periodic retraining; an abrupt structural shift
  in market behavior could outpace it, which is partly why the automated risk controls
  exist.
- **Status.** The system is in **paper trading**; reported numbers are not realized
  returns on capital. A decision gate evaluates the strategy against conservative,
  pre-registered thresholds before any capital is committed.

**Ethics and sustainability.** The model uses only market and on-chain data — no
personal data — so fairness/bias concerns are minimal. Its computational footprint is
very low (CPU-only, retrained twice a year). As a credibility safeguard, the project
adopts an explicit honesty policy: results are always labeled as backtest/paper-trade
and reported with their deflated, conservative range.

---

## 6. Conclusion and next steps

This project demonstrates that a disciplined machine-learning approach can turn a
discretionary "how much Bitcoin to hold" decision into a **systematic, auditable, and
risk-controlled** process that, in out-of-sample testing, substantially improves
downside risk-adjusted returns and sharply reduces drawdown relative to static
allocations. The evidence that the signal is statistically real — surviving permutation
and deflation tests — distinguishes it from an overfit backtest.

Next steps: complete the paper-trading window and pass the decision gate before
committing capital; add model-explainability tooling; and, conditional on the gate,
invest in production hardening and extend the same framework to additional asset/cash
pairs.

---

## References

- F. Sortino and L. Price (1994). *Performance Measurement in a Downside Risk
  Framework.* Journal of Investing.
- M. López de Prado (2018). *Advances in Financial Machine Learning.* Wiley
  (walk-forward, purging/embargo, fractional differentiation).
- D. Bailey and M. López de Prado (2014). *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio
  Management.

---

*Public Report — Group 101 · Module 16 · Inteli (T22, 2025-2A) · Partner: Hashdex.
Export as `Public Report G101 M16.pdf` for submission to the public repository.*
