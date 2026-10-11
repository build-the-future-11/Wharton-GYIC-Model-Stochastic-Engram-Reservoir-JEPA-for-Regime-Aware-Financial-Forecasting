# FREM proposed economic objective and risk ablations

**Status: PROPOSED_NOT_FROZEN / NOT_EVALUATED.** This is the ACT-042 design
proposal for review, based on PR #4 at
`52cb26d582d2529af85a1fb7ce00ab1b89d47b03`. It adds no portfolio implementation,
empirical protocol freeze, data access, training, evaluation or efficacy claim.
The existing [prediction protocol](PROTOCOL_DRAFT_20261010.md), chronological
controls, label embargo and immutable exploratory outputs remain in force.

## 1. Decisions that remain unresolved

No investor profile, competition edition or authority to freeze an empirical
study is inferred from the repository name. Before implementation or evaluation,
the investigator and independent reviewer must resolve and record the following.

| Decision | Required evidence or choice; current status |
| --- | --- |
| Owner and freeze authority | Named investigator, independent reviewer, person authorized to approve a versioned protocol and protected access: **UNRESOLVED**. |
| Investor mandate | Verified client/case edition, base currency, capital, withdrawals, goals, horizon, loss tolerance, concentration and group limits, eligible instruments, cash treatment, shorting and leverage permissions: **UNRESOLVED**. No asset or allocation is recommended here. |
| Data and availability | Licensed point-in-time universe and prices, distributions, delistings, FX, revisions, timestamps, immutable identities, permitted storage/access and protected split custodian: **UNRESOLVED**. |
| Decision and execution clock | Signal cutoff, label maturity, exchange calendars, reference marks, first executable order time, holding interval, rebalance rule and missing/suspended asset handling: **UNRESOLVED**. |
| Costs and size | Commission, full spread, incremental slippage, impact, fees, taxes if applicable, liquidity/participation, rounding, borrowing/funding and cost stress cases: **UNRESOLVED**. No numerical cost assumption is adopted by the examples below. |
| Statistical design | Primary contrast and endpoint, risk level, threshold, effect/noninferiority margins, family of ablations, uncertainty method, sample sufficiency, development budget and failure rules: **UNRESOLVED**. |

The official [Wharton case page](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/case-study/)
is a source to verify the applicable mandate; this document does not supply a
client mandate or certify competition compliance.

## 2. Return accounting and the proposed reward

The economic target would be a **one-interval total return in the chosen base
currency**, including distributions and consistent corporate-action treatment.
This proposed target does not change the current data-agnostic prediction CLI
or settle its price-versus-return choice. Predictive error alone cannot establish
an economic benefit.

Let the information set at the order cutoff contain only information actually
available then. Orders may execute only at the declared eligible time after
their inputs become available: a closing-price signal cannot receive an earlier
or simultaneous close fill without a separately justified executable mechanism.
Use one common reference-mark convention across assets and methods. The return
and cost equations below describe the **realized accounting**, not permission
to use realized fills or future costs when choosing orders.

For a candidate cash-funded formulation, let:

- $V_{t^-}>0$ be wealth immediately before trading, at the common reference marks;
- $\widetilde w_{i,t}$ be the drifted pretrade risky holding divided by $V_{t^-}$;
- $w_{i,t}$ be the post-trade risky holding at those marks, also divided by
  **pretrade** wealth, and $\Delta_{i,t}=w_{i,t}-\widetilde w_{i,t}$;
- $T_t=\sum_i|\Delta_{i,t}|$ be gross traded notional divided by pretrade wealth.
  This definition has no implicit factor of one half;
- $C_t\geq0$ be execution costs paid at trading, divided by pretrade wealth;
- $c_t=1-\sum_iw_{i,t}-C_t$ be the remaining cash fraction;
- $r_{i,t+1}$ and $r_{c,t+1}$ be asset total returns and cash return from these
  marks to the next pretrade boundary, without counting the same execution costs
  again inside the marked asset return.

For that formulation, the self-financing identity is

$$
R^{\mathrm{net}}_{t+1}=\sum_iw_{i,t}r_{i,t+1}+c_t r_{c,t+1}-C_t,
\qquad V_{(t+1)^-}=V_{t^-}(1+R^{\mathrm{net}}_{t+1}).
$$

Cash-funded nonnegative holdings are the scope of this algebra, **not an assumed
investor constraint**. Shorting, leverage, derivatives, securities lending and
financing remain unresolved; admitting them requires an explicitly reviewed
extension for collateral, signed cash rates and fees. Post-cost normalized
portfolio weights must not be substituted for the pretrade-normalized $w$ above.
If foreign assets are later admitted, define FX as base-currency units per local
unit and use $1+r^{\mathrm{base}}=(1+r^{\mathrm{local}})(1+r^{\mathrm{FX}})$.

A proposed execution-cost decomposition, with all parameters still unresolved,
is

$$
C_t=\sum_i\left[(a_{i,t}+s_{i,t}/2+m_{i,t})|\Delta_{i,t}|
 +\frac{f_{i,t}}{V_{t^-}}\mathbf1_{\{\Delta_{i,t}\ne0\}}\right]
 + I_t(\Delta_t).
$$

Here $a$ is proportional commission, $s$ the **full** relative bid-ask spread,
$m$ incremental slippage beyond that spread, $f$ a fixed fee in base-currency
units and $I_t\geq0$ a separately specified impact fraction. All rates are decimal
fractions, not basis-point numbers. Require $a,s,m,f\geq0$ and pure trade impact
$I_t(0)=0$. Any custody, idle-account or other nontrading fee needs a separate,
explicitly timed cashflow term and a corresponding extension of the wealth
identity; it must not be hidden in trade impact. A fill-price convention that already charges
spread/slippage must not also charge those terms here. Expected costs used at
order time and realized costs used for accounting must be separately recorded;
realized fills cannot retroactively change the signal or chosen order. Costs
are charged on the same execution schedule for every method, including initial
entry and terminal liquidation if those events belong to the future declared
endpoint. Any permitted tax treatment must also be specified before claiming a
net investor return. Zero-cost results could only be labeled a diagnostic.

The return term in the proposed decision objective is
$\mu_t(w)=\mathbb E[R^{\mathrm{net}}_{t+1}(w)\mid\mathcal I_t]$, with uncertainty
about fills/costs included if relevant. Let a proposed per-interval shortfall
threshold be $\tau_t$ and define
$L_{t+1}(w)=(\tau_t-R^{\mathrm{net}}_{t+1}(w))_+$. Threshold, confidence level
$0<\alpha<1$ and penalty strengths are review decisions, not selected values.

Use the following variational tail-loss functional, including for a discrete
scenario distribution, rather than an ambiguous average of observations
strictly exceeding a quantile:

$$
\operatorname{CVaR}_{\alpha,t}(L)=\min_{z\in\mathbb R}
 \left[z+\frac{\mathbb E[(L-z)_+\mid\mathcal I_t]}{1-\alpha}\right].
$$

The formulation follows [Rockafellar and Uryasev, *Optimization of Conditional
Value-at-Risk*, equations (4)–(5)](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf).
The extension beyond continuous losses is treated in their companion paper,
[*Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf).
Using the displayed functional specifies the tail convention when losses have
atoms; it is a proposed modeled-risk criterion, not a certification of actual
tail coverage.

The candidate economic score is

$$
J_t(w)=\mu_t(w)-\lambda_{\mathrm{tail}}\operatorname{CVaR}_{\alpha,t}(L(w))
 -\lambda_{\mathrm{turn}}T_t(w)-\lambda_{\mathrm{conc}}\sum_iw_i^2,
\qquad w\in\mathcal F_t.
$$

All penalties are nonnegative. Return and tail loss are fractions per the same
interval; $\lambda_{\mathrm{tail}}$ is dimensionless. Turnover and concentration
are dimensionless, so their coefficients carry return-fraction units for that
interval. The turnover penalty expresses an additional preference and is not a
second transaction-cost deduction. Neither an annualized Sharpe ratio nor a
lifetime drawdown is a one-interval reward in this specification. Compounded
wealth and realized drawdown would be separate path summaries; this objective
does not guarantee a long-term goal or maximum drawdown.

## 3. Position sizing and fair execution

The candidate position rule would maximize the reviewed score over a common
feasible set $\mathcal F_t$. Its asset eligibility, cash reserve, gross/net
exposure, per-name and group bounds, turnover cap, liquidity/participation and
lot constraints all require the missing mandate and execution specification.
No numerical limit is frozen here. Sizing must use information and a declared
cost model available before the order, reserve cash for costs, and recheck the
budget and hard bounds after any lot rounding. Reconcile actual fills and cash
without rewriting the order decision.

All arms receive the same decision times, eligible universe, capital, execution
assumptions and **external hard risk constraints**. Turning a statistical risk
penalty off does not remove an investor mandate or relax a trading constraint.
The optimization solver, tolerances, deterministic tie handling and treatment
of infeasibility, missing prices, partial fills and order failures must be
declared together. They must retain every scheduled date and failure; a silent
switch to cash or deletion of a difficult date is not an admissible default.

## 4. Separate the component and downside-risk questions

An economic comparison must separate the return predictor, the downside model
and the common position rule. A return head estimates conditional return;
a downside/scenario head estimates the loss distribution needed for the risk
term. They may use only mature training targets. Their architectures, calibration
and training losses remain proposed work; an unvalidated uncertainty output is
not calibrated risk. Independently sampled marginal asset returns must not be
treated as a joint tail distribution without a reviewed dependence model.

| Comparison family | Required contrasts and interpretation |
| --- | --- |
| Conventional forecasts | Existing historical mean, lag persistence and raw Ridge; prospective strong conventional alternatives selected before protected access. No unimplemented XGBoost result is assumed. |
| Fixed components | Preserve the existing full fixed Fourier/reservoir/sign-bigram readout and component removals. Match information, chronology, readout selection and declared development budget. |
| JEPA-style development component | Compare the optional prototype with its matched no-JEPA counterpart once prospectively authorized. Its current artificial software checks do not establish learned market states or alpha. |
| True episodic memory | Engram-on versus matched engram-off is **pending implementation and versioning**. A sign-bigram feature is not evidence of episodic retrieval; no existing engram efficacy is inferred. |
| Return-only allocator | Use the same return forecasts, common feasible set and cost treatment with $\lambda_{\mathrm{tail}}=0$; retain any common non-tail penalties. |
| Conventional risk versus learned downside | Hold return forecasts fixed. Compare the same risk functional fed by a declared conventional historical joint scenario/risk estimate versus the proposed learned joint downside estimate, with identical $\alpha$, $\tau$, penalty rule and hard constraints. This isolates the risk estimator. |
| Downside contribution | Compare the learned-risk arm with its tail penalty removed, using identical return forecasts and all other common settings. This isolates use of the tail term, not a different return head. |
| Simple allocation controls | Distinct prospective all-cash and carry-forward/no-rebalance controls; eligible equal-weight and conventional historical inverse-volatility/risk-based allocations, where the mandate admits them. Declare initial holdings and any initial conversion or terminal liquidation, charging each actual trade. Carry-forward retains and drifts its specified starting holdings; it is not silently treated as cash. Universe and rule details remain unresolved. |

The reviewer must choose a bounded primary family before freezing; this table
does not authorize an exhaustive search. Capacity, feature availability,
hyperparameter-selection budget, training/inference cost and refit frequency
must be disclosed for trainable comparisons. Simple no-fit controls remain
valid and must not be weakened to match wasted compute. A component removal
that changes capacity or optimization must be identified as such, with a
capacity-matched control where feasible. Forecast ablations are not evidence
about the economic allocator until that separate comparison is authorized.

## 5. Falsification and no-lookahead boundaries

Two candidate claims require separate prospective contrasts:

1. **Return benefit:** net return improves on the declared strongest conventional
   comparator by a practical margin, subject to a declared risk noninferiority
   requirement. Endpoint, margin and uncertainty rule are unresolved.
2. **Downside benefit:** the learned-risk arm reduces the declared tail-loss
   endpoint relative to the conventional-risk arm with an explicitly bounded
   return sacrifice. The risk margin and permissible sacrifice are unresolved.

A claim is unsupported if its interval fails the predeclared superiority or
noninferiority rule. A sufficiently precise opposite result counts against the
corresponding claim in the tested setting. Benefits erased by the approved cost
cases, conventional controls that dominate, or no incremental component benefit
must be retained. Wide uncertainty is inconclusive. Broken provenance,
lookahead, invalid executions or protocol violations invalidate the affected
comparison; they are not scientific null results.

Before any empirical freeze, declare licensed point-in-time data identities and
universe membership, target units, calendars and actual release/availability
times. Corporate-action and revision handling must not leak future information.
All preprocessing, feature fitting, risk calibration, retrieval construction,
penalty/threshold selection and model choice belong to the specified training
and development windows. A supervised memory may acquire a label only after
that label matures. A row-count embargo alone cannot certify real-world release
timing. Existing first-origin label embargo and rolling-origin rules remain
unchanged; allowing previously observed values at later origins is not permission
to revisit protected outcomes for selection.

The future design must fix training/development/protected roles, overlap and
purging rules, missing/delisted-asset handling, primary contrasts, cost cases,
effect thresholds and failure accounting before protected access. Economic
comparisons should pair the same eligible calendar periods. Uncertainty must
respect serial dependence and shared/overlapping returns; model seeds are not
independent market samples. The block/resampling unit, interval method and
minimum informative sample size remain review decisions.

Keep gross and net returns, the cost decomposition, turnover, cash, gross/net
exposures, constraint and execution failures, realized tail-loss summaries and compounded path summaries
for every predeclared arm and period. Model-based ex-ante CVaR and an ex-post
tail estimate are different quantities and must be labeled separately. Lower
risk caused by more cash, smaller exposures or less trading is an allocation
effect; it does not by itself show an improved forecasting mechanism. Report
all prespecified contrasts, including negative and inconclusive results, without
selecting a profitable regime or seed. No such outputs were read or produced
for this document.

## 6. Analytic checks of units and signs

These are constructed arithmetic examples, not data, predictions, parameter
choices, risk estimates or a backtest. Cash earns zero only in these examples.

| Example | Calculation and expected value |
| --- | --- |
| Entry costs and cash | With wealth 1,000, risky fraction changing from 0.4 to 0.5, and illustrative commission 10 bp = 0.001: $T=0.1$, $C=0.0001$ (1 bp of wealth = 0.10 currency unit), $c=0.4999$. If the marked asset return is 0.02, net return is $0.5(0.02)-0.0001=0.0099$ (0.99%); ending wealth is 1,009.90. |
| Two-sided reallocation | Moving risky fractions $(0.4,0.6)$ to $(0.5,0.5)$ trades gross notional 0.2; the same illustrative rate gives cost 0.0002 = 2 bp. The intended fully invested weights would require extra cash and are **infeasible as written** for the cash-funded formulation; sizing must reserve the cost, not report an unfinanced trade as executable. |
| Full spread convention | A full spread of 20 bp and gross traded fraction 0.1 produces $0.002/2\times0.1=0.0001$ = 1 bp of wealth, before any other cost. |
| FX compounding | A local return of 2% and a base-per-local FX return of 1% give $1.02\times1.01-1=0.0302$ = 3.02%. |
| Tail convention | Four equally weighted hypothetical shortfalls $(0,0,0,0.04)$ have mean 0.01. At illustrative $\alpha=0.75$, the displayed variational functional is 0.04, including the quantile-boundary mass correctly. |
| Penalty sign | With expected return 0.01, modeled tail loss 0.04 and illustrative $\lambda_{\mathrm{tail}}=0.5$, the score before other penalties is $0.01-0.5(0.04)=-0.01$. Reducing that tail loss to 0.02 raises the score to 0 while holding return fixed. |

Only these deterministic calculations and documentation preservation were
checked for this addition. They establish no software implementation, calibrated
tail model, investor suitability, scientific completion or market advantage.
