# Machine Learning for Dynamic Bitcoin Allocation: A Risk-Adjusted, Long-Only Strategy

**Final Course Project (Public Report) — Module 16**
Submitted to the Institute of Technology and Leadership (INTELI), Corporate Track.

**Group 101 (G101)** · Class T22 · 2025-2A · São Paulo
**Business partner:** Hashdex · **Advisor:** Prof. [advisor name]

> **Confidentiality note.** This is a public document. Confidential partner
> information — assets under management, client data, real positions, and
> proprietary implementation details (the exact feature set and hyperparameters) —
> has been intentionally omitted. Performance figures are **out-of-sample backtest
> and early paper-trading results, not realized returns on capital.** Monetary
> values used in the impact analysis are **fictitious and illustrative.**

---

## Resumo

Investidores institucionais brasileiros enfrentam um dilema ao buscar exposição a
Bitcoin (BTC): o ativo oferece prêmio de risco no longo prazo, mas com quedas
(drawdowns) de 60% a 80% que são incompatíveis com a maioria dos mandatos. Este
trabalho, desenvolvido com a gestora de criptoativos Hashdex, apresenta uma solução
de aprendizado de máquina que decide semanalmente qual fração de uma carteira manter
em BTC versus um indexador de caixa (o CDI), de forma **long-only** (0% a 100%, sem
venda a descoberto) e otimizada para **retorno ajustado ao risco de queda** (Sortino).
O núcleo é um ensemble duplo de árvores de decisão com gradient boosting (XGBoost) que
prevê o retorno do BTC em três dias; a previsão é convertida em uma alocação
dimensionada por um filtro de regime de mercado e pela confiança do modelo, e limitada
por controles de risco automáticos (kill switch, redução por acurácia e monitor de
drift). A validação seguiu práticas de finanças quantitativas: walk-forward
out-of-sample com purga temporal, teste de permutação de rótulos, Sharpe deflacionado
e validação com múltiplas sementes. Em um backtest de 4,28 anos out-of-sample, a
estratégia obteve Sortino diário de 3,53 e drawdown máximo de -7,1%, contra -21,7% de
uma alocação estática 30% BTC/70% CDI e -66,7% de uma posição comprada em BTC. O sinal
é estatisticamente real (nenhuma de 100 permutações superou o baseline; p < 0,01). A
solução encontra-se em fase de paper trade, com um gate de decisão de capital previsto
para o terceiro trimestre de 2026.

**Palavras-chave:** aprendizado de máquina; alocação de ativos; bitcoin; gestão de
risco; aprendizado de máquina em finanças.

---

## Abstract

Brazilian institutional investors face a dilemma when seeking Bitcoin (BTC) exposure:
the asset offers a long-run risk premium but with 60%–80% drawdowns that are
incompatible with most mandates. This project, developed with the crypto asset manager
Hashdex, presents a machine-learning solution that decides weekly what fraction of a
portfolio to hold in BTC versus a cash benchmark (the Brazilian interbank rate, CDI),
in a **long-only** manner (0% to 100%, no short selling) and optimized for **downside
risk-adjusted return** (the Sortino ratio). The core is a dual ensemble of
gradient-boosted decision trees (XGBoost) that forecasts the three-day forward BTC
return; the forecast is converted into an allocation scaled by a market-regime filter
and the model's confidence, and bounded by automated risk controls (kill switch,
accuracy de-risk, and a feature-drift monitor). Validation followed quantitative-finance
best practice: walk-forward out-of-sample testing with temporal purging, label
permutation testing, the Deflated Sharpe Ratio, and multi-seed validation. In a
4.28-year out-of-sample backtest the strategy achieved a daily Sortino ratio of 3.53
and a maximum drawdown of -7.1%, versus -21.7% for a static 30% BTC / 70% CDI
allocation and -66.7% for a buy-and-hold BTC position. The signal is statistically real
(none of 100 label permutations beat the baseline; p < 0.01). The solution is in a
paper-trading phase, with a capital-allocation decision gate planned for the third
quarter of 2026.

**Keywords:** machine learning; asset allocation; bitcoin; risk management; financial
machine learning.

---

## 1. Introduction

### 1.1 Partner company context

Hashdex is an asset manager specialized in crypto assets, operating index products and
investment vehicles for institutional and retail investors. The relevant area for this
project is the quantitative/portfolio-management desk, responsible for deciding how
much exposure to risky crypto assets a given vehicle should carry over time.

### 1.2 Motivation

Bitcoin allocation decisions are frequently **discretionary** — hard to audit, to
reproduce, and to scale across vehicles or review by a risk committee. A systematic,
auditable process that captures most of Bitcoin's upside while sharply limiting its
drawdown would be strategically valuable: it reduces key-person risk, standardizes a
recurring decision, and is defensible to investors and regulators.

### 1.3 Problem definition

The core problem is to **capture Bitcoin's risk premium with a fraction of its
drawdown**, systematically. The baseline (the current alternative) is a static
allocation. Over the 2022–2026 test window, in BRL: a 100% BTC buy-and-hold position
returned +15.5% CAGR but suffered a **-66.7% maximum drawdown**; a static 30% BTC / 70%
CDI blend returned +17.0% CAGR with a -21.7% drawdown and a Sortino ratio of 1.13; cash
(CDI) returned +13.0% with no drawdown. The opportunity is to dominate these baselines
on a downside-risk-adjusted basis.

### 1.4 Proposed solution and expected contribution

We propose a computational pipeline that, each week, generates a target BTC/CDI
allocation from a machine-learning forecast, scaled by market regime and model
confidence and bounded by automated risk controls. The **contribution objective** is a
strategy that, out-of-sample, materially improves the Sortino ratio and reduces maximum
drawdown relative to static allocations, while remaining long-only.

### 1.5 Business objectives

For the partner, the expected results are: (i) a systematic, auditable allocation
process that replaces discretionary judgment; (ii) superior downside risk-adjusted
performance (target Sortino ≥ 2.5; conservative floor ≥ 1.5 after deflation);
(iii) maximum drawdown materially below buy-and-hold; and (iv) an operationally
reliable signal validated in paper trading before any capital is committed.

### 1.6 Structure of this report

Section 2 develops the solution: the applied rationale (2.1), the specification and
development of the pipeline (2.2), and the assessment of impact and contribution to the
business (2.3). Section 3 concludes and outlines next steps.

---

## 2. Solution development

### 2.1 Applied rationale

**Business-area rationale.** In crypto asset management, the dominant risk is the
asymmetry of Bitcoin's return distribution — large, infrequent gains alongside deep,
prolonged drawdowns. Market best practice for managing such exposure ranges from static
rebalancing to discretionary market timing; both leave risk-adjusted return on the
table. A systematic, regime-aware allocation that prioritizes downside protection is
the gap this project addresses.

**Technological rationale.** A fixed rule (e.g., a permanent 30% BTC sleeve, or a pure
moving-average crossover) does not adapt to regime and cannot size a position by the
strength and confidence of a view. A supervised machine-learning model can. We adopt
**gradient-boosted decision trees (XGBoost)** because, on tabular data with roughly 30
engineered features and a few thousand observations, gradient boosting empirically
outperforms recurrent neural networks, random forests, and stacked meta-learners — all
of which were tested and rejected on the target objective. Bagging many trees per
ensemble reduces seed-to-seed variance and stabilizes the live signal.

**Foundations from financial-ML methodology.** The design incorporates established
literature: the Sortino ratio (Sortino and Price, 1994) as an objective that penalizes
only downside volatility — appropriate because Bitcoin's fat tails are predominantly on
the upside; walk-forward validation with temporal **purging/embargo** and fractional
differentiation from López de Prado (2018); the **Deflated Sharpe Ratio** (Bailey and
López de Prado, 2014) to discount multiple-testing; and fractional-Kelly position
sizing for the confidence scaling.

### 2.2 Specification and development

**Data.** The pipeline ingests roughly a dozen public and licensed sources spanning
three families — macroeconomic (e.g., monetary aggregates, policy-rate proxies),
on-chain (network and valuation metrics), and market/technical (price, volume,
derivatives basis and funding). From these it derives about thirty features. Feature
construction is strictly **backward-looking**, a property enforced by an automated
property test that perturbs the most recent observation and verifies that no past
feature value changes — guarding against look-ahead bias.

**Model and sizing.** Two bagged XGBoost ensembles are trained: a regressor for the
three-day forward BTC return and a classifier for its direction. The regression
forecast is converted into a target allocation that is **scaled by two factors**: a
market-regime filter (derived from long- and short-term moving averages) that reduces
exposure in bearish regimes, and a **confidence weight** from the directional
classifier, so the strategy bets larger only when the model is more certain. The result
is clipped to the [0%, 100%] long-only range. The portfolio rebalances weekly, with an
extra emergency rebalance on very large single-day moves, and the model is retrained on
a fixed semi-annual schedule using an expanding window.

**Risk controls.** Three automated controls are applied to every signal and are part of
the production code: a **kill switch** that caps exposure when cumulative drawdown
breaches a threshold; an **accuracy de-risk** that halves exposure when rolling
predictive accuracy deteriorates while the model remains overconfident; and a
**population-stability monitor** that flags feature-distribution drift.

**Validation methodology.** The system is evaluated with walk-forward, out-of-sample
testing (expanding window, temporal purge/embargo), multi-seed runs to quantify metric
variance, label-permutation testing to establish a null distribution, and the Deflated
Sharpe Ratio to discount the number of configurations tried.

**Engineering.** The solution is a one-command production pipeline with pinned
dependencies for reproducibility, a configuration fingerprint that detects a stale model
and forces retraining, and a test suite of 132 tests (covering look-ahead, determinism,
train/serve parity, and the risk controls).

### 2.3 Assessment of impact and contribution to the business

All figures below are **out-of-sample backtest** results over 4.28 years (248 weekly
rebalances), reported on a multi-seed basis and aware of realistic transaction costs (a
few basis points per rebalance; results remain strong even under a pessimistic 50 bps
assumption).

**Table 1 — Risk-adjusted performance (4.28-year out-of-sample backtest, BRL).**

| Strategy | CAGR | Sortino (daily) | Max drawdown |
|---|---|---|---|
| Cash benchmark (CDI) | +13.0% | — | 0% |
| Static 30% BTC / 70% CDI | +17.0% | 1.13 | -21.7% |
| 100% Bitcoin (buy-and-hold) | +15.5% | 0.56 | -66.7% |
| **This strategy (long-only ML)** | **+57.3%** | **3.53** | **-7.14%** |

The strategy delivered markedly higher downside-risk-adjusted performance than either
static allocation, holding maximum drawdown to roughly **one-tenth** of buy-and-hold — a
direct consequence of spending most of the time (~58%) in cash and concentrating
exposure (~15% average) in favorable regimes.

**Statistical significance.** In label-permutation testing, **none of 100 shuffled-target
runs beat the baseline** (p < 0.01). A feature-composition probe shows that signal +
regime + confidence alone yield a Sortino of 2.19, while the ML magnitude forecast adds
**+3.4** — i.e., the model is the source of the incremental edge, not a fixed rule.
After applying a Deflated Sharpe Ratio for multiple-testing, the strategy still passes
under a realistic estimate of the number of effective trials.

**Early live (paper) evidence.** In a strict out-of-sample window of the most recent
year-to-date (105 days, 17 rebalances), the strategy returned **+19.67%** while Bitcoin
buy-and-hold returned **-14.36%** and cash **+4.09%** — early corroboration, though
concentrated in a small number of rebalances.

**Business impact.** For an illustrative (fictitious) allocated capital, the value
created is the incremental return over the static baseline; because the operating cost
of the solution is negligible relative to that incremental return, the decisive variable
for return on investment is not cost but **whether the edge persists live** — precisely
what the paper-trading phase and the decision gate exist to answer. The qualitative
contribution is a systematic, auditable allocation process that reduces operational and
key-person risk.

---

## 3. Conclusion

This project demonstrates that a disciplined machine-learning approach can turn a
discretionary "how much Bitcoin to hold" decision into a **systematic, auditable, and
risk-controlled** process that, in out-of-sample testing, substantially improves
downside risk-adjusted returns and sharply reduces drawdown relative to static
allocations. The evidence that the signal is statistically real — surviving permutation
and deflation tests — distinguishes it from an overfit backtest.

We are deliberately conservative about the results. After discounting for
multiple-testing, the realistic forward expectation is a Sortino in the 1.5–2.5 range
and a drawdown of -15% to -25% — still superior to the static baselines, but well below
the raw backtest. The return is also concentrated in a few decisive weeks, and the model
depends on periodic retraining, which is why automated risk controls exist.

**Next steps:** complete the paper-trading window and pass the decision gate before
committing capital; add model-explainability tooling (feature attribution) to the
production signal; and, conditional on the gate, invest in production hardening and
extend the same framework to additional asset/cash pairs. The model uses only market and
on-chain data (no personal data), its computational footprint is low (CPU-only,
retrained twice a year), and the project adopts an explicit honesty policy: results are
always labeled as backtest/paper-trade and reported with their deflated, conservative
range.

---

## References

BAILEY, D. H.; LÓPEZ DE PRADO, M. The Deflated Sharpe Ratio: correcting for selection
bias, backtest overfitting, and non-normality. **The Journal of Portfolio Management**,
v. 40, n. 5, p. 94–107, 2014.

CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. In: **Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining**.
New York: ACM, 2016. p. 785–794.

LÓPEZ DE PRADO, M. **Advances in Financial Machine Learning**. Hoboken: John Wiley &
Sons, 2018.

SORTINO, F. A.; PRICE, L. N. Performance measurement in a downside risk framework.
**The Journal of Investing**, v. 3, n. 3, p. 59–64, 1994.

---

*Public Report — Group 101 · Module 16 · INTELI (T22, 2025-2A) · Partner: Hashdex.
Export as `Public Report G101 M16.pdf` for the public repository.*
