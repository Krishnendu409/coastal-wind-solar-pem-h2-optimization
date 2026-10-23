# Publication and Strategic Presentation Strategy

**Paper Title:** Physics-Guided Surrogate Optimization and Load-Compensated Multivariate Fault Monitoring of a Coastal Hybrid Wind–Solar–PEM Green Hydrogen Plant under Multi-Year Meteorological Uncertainty  
**Authors:** Krishnendu (B.Tech CSE, MIT Manipal) & Arav Aggarwal (B.Tech Civil Engineering, MIT Manipal)  
**Date:** October 2026  

---

## 1. Target Publication Roadmap

### 1.1 Immediate Target: CHEMIGNITE 2026 (ASPIRE)
- **Event:** National Chemical Engineering Research Paper Presentation & Competition
- **Host Institution:** Department of Chemical Engineering, Manipal Institute of Technology (MIT), MAHE, Manipal
- **Key Dates:** Abstract Submission: October 25, 2026 | In-Person Presentation: October 29, 2026 (Sir M. Vishveshwaraiah Seminar Hall, Academic Block 2)
- **Evaluator Alignment:**
  - **Chief Guest & Evaluator: Prof. Kishalay Mitra (IIT Hyderabad, Head of GOKUL Lab):** Renowned expert in Green Deep Learning surrogates, Explainable AI, multi-objective NSGA-II optimization, and high-current PEM water electrolysis (author of Masampally, Mitra & Runkana, *Energy Convers. Manage.* 2026, and Pujari, Inapakurthi & Mitra, *Comput. Chem. Eng.* Aug. 2026).
  - **Internal Evaluator: Dr. Muddu Madakyaru (Professor, Chemical Engineering, MIT Manipal):** Leading expert in Process Systems Engineering (PSE) and Multivariate Statistical Process Monitoring (MSPM) via PCA, Hotelling's $T^2$, and Jackson–Mudholkar $Q/\text{SPE}$ control charts for nonlinear chemical processes.

### 1.2 Top-Tier Journal Target Options (Post-Conference Expansion)

| Journal | Publisher | Impact Factor | Core Scope & Strategic Fit | Target Section / Category |
| :--- | :---: | :---: | :--- | :--- |
| ***Computers & Chemical Engineering*** | Elsevier | 4.3 | Process systems engineering, surrogate-assisted optimization, multi-objective evolutionary algorithms, Green AI. | Full Research Article (Primary Target) |
| ***Energy Conversion and Management*** | Elsevier | 10.4 | Electrochemical energy systems, high-current PEM water electrolysis, multi-energy techno-economics. | Full Research Article (High-Impact Target) |
| ***International Journal of Hydrogen Energy*** | Elsevier | 7.2 | Hydrogen production, water electrolysis modeling, degradation mechanisms, gas crossover safety. | Specialized Hydrogen Article |
| ***IEEE Transactions on Sustainable Energy*** | IEEE | 8.6 | Multi-carrier renewable microgrids, wind-solar integration, grid open-access wheeling, fault monitoring. | IEEE Transactions Paper |

---

## 2. Anticipating & Defending Against Peer Review Challenges

### Reviewer Challenge 1: "Why use a machine learning surrogate when a 1-year model runs in 157 ms?"
- **Anticipated Critique:** Reviewers often question the computational necessity of surrogate models for simple zero-dimensional energy balance models that run in a fraction of a second.
- **Defensible Response:** 
  1. *Climatological Horizon:* A single-year 8,760-hour model is insufficient because it ignores severe inter-annual monsoon variability and multi-year climate risks. Our framework evaluates designs over a contiguous 6-year ($52,608$-hour) NASA POWER MERRA-2 record.
  2. *Evolutionary Multi-Seed Search:* Executing a 5-seed constrained NSGA-II optimization ($160$ population $\times$ $80$ generations $\times$ $5$ seeds $= 64,000$ evaluations) on a full 52,608-hour dynamic differential-algebraic simulation requires $\sim 4.5\text{ CPU-hours}$ even with Numba JIT acceleration (or over $7.2\text{ CPU-days}$ in pure Python ODE).
  3. *Surrogate Efficiency:* The proposed `Green-MLP` evaluates candidate designs in $8.55\ \mu\text{s}$, enabling the full 64,000-evaluation search in less than $0.6\text{ seconds}$ ($4,206\times$ acceleration) while preserving near-perfect Pareto rank correlation ($\rho = 0.9994$) and achieving an LCOH RMSE of only $\text{INR } 1.19/\text{kg}$ on the held-out Pareto front.

### Reviewer Challenge 2: "Where is the experimental validation for the polarization curve and the wind resource?"
- **Anticipated Critique:** Many simulation papers present unvalidated, assembled models with arbitrary empirical multipliers.
- **Defensible Response:**
  1. *Electrochemical Validation:* We explicitly calibrated the model against $N = 14$ experimental polarization data points ($0.10\text{–}3.0\text{ A/cm}^2$) from Bernt & Gasteiger (*J. Electrochem. Soc.* 2016) and Rakousky et al. (*J. Power Sources* 2016) on Nafion 115 ($127\ \mu\text{m}$, $75^\circ\text{C}, 30\text{ bar}$), achieving an exact fit ($\text{RMSE} = 3.10\text{ mV}$, $R^2 = 0.99971$).
  2. *Wind Resource Integrity:* We removed all empirical wind speed multipliers ($f_{\text{wind}} = 1.0$), establishing that on-site coastal wind CF is $14.70\%$. We then modeled green open-access wheeled wind from Karnataka's Chitradurga–Gadag highland corridor ($43.29\%$ delivered CF) via the $220\text{ kV}$ state transmission grid based on published KERC Open Access Tariff Orders ($₹0.58/\text{kWh}$).

### Reviewer Challenge 3: "Does high-current overload ($i_{\max} > 2.0\text{ A/cm}^2$) truly lower LCOH?"
- **Anticipated Critique:** Sizing literature often assumes that operating at higher current densities uniformly reduces capital expenditure by shrinking the stack cell area.
- **Defensible Response (Our Nuanced Scientific Discovery):**
  1. *Split-CAPEX Reality:* While stack active cell area decreases with higher current density, power electronics, rectifiers, transformers, and thermal/gas balance-of-plant ($42\%$ of total electrolyzer capital cost) must be sized for peak electrical power throughput $P_{\text{PEM,peak}}$.
  2. *Economic Penalty:* Sizing BOP for peak megawatt overload costs $\text{INR } 2.436\text{ Cr/MW}$, which outweighs the cell-area savings, while concurrently increasing specific energy consumption from $52.62$ to $58.51\text{ kWh/kg}$.
  3. *Optimal Design:* As a result, the NSGA-II knee point operates at $i_{\max}^* = 1.764\text{ A/cm}^2$ (below nominal $1.80\text{ A/cm}^2$). Sizing at fixed nominal current achieves $\text{INR } 329.12/\text{kg}$ ($1.03\%$ cheaper), refuting the naive assumption that high-current overload always reduces levelized cost.

### Reviewer Challenge 4: "Does PCA actually detect faults earlier than univariate alarms?"
- **Anticipated Critique:** Prior claims of dramatic early warning lead times (e.g., 52 hours) often rely on straw-man comparisons against static, uncompensated voltage limits under dynamic renewable power.
- **Defensible Response (Scientific Honesty):**
  1. *Fair Baseline:* We benchmarked 9-sensor PCA against a load-compensated 1D voltage-residual EWMA chart ($r_V(t) = V_{\text{meas}}(t) - V_{\text{model}}(i(t), T(t))$) across 100 independent Monte Carlo realizations with sensor noise.
  2. *Detection Latency Reality:* The load-compensated 1D EWMA filter detects incipient membrane dehydration in $10.6 \pm 3.6\text{ h}$ with a lower false alarm rate ($5.39\%$), whereas PCA detects it in $11.6 \pm 3.2\text{ h}$ ($\text{FAR} = 9.80\%$).
  3. *PCA's True Value:* PCA does not provide an early detection lead time over a load-compensated residual. Its true, indispensable value is **multi-sensor root-cause diagnosis**. While 1D EWMA cannot identify what caused a voltage deviation, PCA variable contribution plots isolate Stack Temperature ($60.17\%$) and Cathode Pressure ($35.07\%$) for Fault 1, versus Cell Voltage ($34.61\%$) and Cathode Pressure ($26.04\%$) for Fault 2, directing maintenance engineers directly to the degraded subsystem.

---

## 3. CHEMIGNITE 2026 Presentation Execution Strategy (29 October 2026)

### Presentation Structure (12 Minutes + 3 Minutes Q&A)
1. **Slide 1: Title & Regional Coastal Context (1.5 min):** Siting green hydrogen in the Udupi–Mangalore coastal industrial corridor (MRPL, MCF) under Indian National Green Hydrogen Mission targets ($5\text{ MMT/yr}$).
2. **Slide 2: Regional Climatology & Multi-Year Complementarity (2 min):** Coastal solar PV vs.\ wheeled highland wind (2019–2024, 52,608 hours NASA POWER). Presenting Regime A (pure coastal) vs.\ Regime B (wheeled wind) with KERC open-access wheeling charges.
3. **Slide 3: Calibrated First-Principles PEMWE Electrochemistry (2.5 min):** Exact Nernst thermodynamics ($1.234\text{ V}$), Butler–Volmer activation, Springer ohmic, bubble crowding, and experimental calibration against Bernt & Gasteiger (2016) ($\text{RMSE} = 3.10\text{ mV}$, $R^2 = 0.99971$). Turndown safety with catalytic recombination ($y_{\text{H}_2} \le 1.64\text{ mol\%}$).
4. **Slide 4: Leak-Free Green Deep Learning Surrogate & Ablation (2 min):** $2 \times 5$ factorial ablation. Green-MLP ($4,612$ parameters, $8,784$ FLOPs) achieving $89.94\%$ FLOP reduction, $4,206\times$ speedup, and $\rho = 0.9994$ rank preservation on the Pareto front.
5. **Slide 5: Sizing Results, Split-CAPEX Trade-Off & Monte Carlo Risk (2.5 min):** Knee-point design ($₹332.51/\text{kg}$, $\text{LPSP} = 0.62\%$). Explaining why high-current overload does not reduce LCOH due to peak BOP scaling ($42\%$ of electrolyzer CAPEX). True $N=5,000$ Monte Carlo distribution ($73.3\% < ₹350/\text{kg}$) and Sobol sensitivity.
6. **Slide 6: Load-Compensated MSPM & Fault Diagnosis (1.5 min):** 9-sensor PCA ($k=3$, $96.03\%$ variance) vs.\ load-compensated 1D EWMA. Emphasizing root-cause isolation via contribution plots.
7. **Slide 7: Conclusion & Actionable Roadmap (1 min):** Key findings and policy implications for coastal green hydrogen hubs in Karnataka.

### Evaluator-Specific Engagement
- **For Prof. Kishalay Mitra:** Highlight the computational elegance of the leak-free Green-MLP surrogate, the exact FLOP reduction, the 5-seed constrained NSGA-II hypervolume convergence, and the thermodynamic rigor of the high-current polarization calibration extending his 2026 formulations.
- **For Dr. Muddu Madakyaru:** Highlight the latent-variable PCA formulation on real plant dispatch, Hotelling's $T^2$ and Jackson–Mudholkar $Q$ control limits, the honest benchmark against load-compensated EWMA, and the multi-sensor contribution plots diagnosing membrane dehydration and PTL flooding.
