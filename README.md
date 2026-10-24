# Physics-Guided Surrogate Optimization and Load-Compensated Multivariate Fault Monitoring of a Coastal Hybrid Wind–Solar–PEM Green Hydrogen Plant under Multi-Year Meteorological Uncertainty

[![Paper: IEEE Format](https://img.shields.io/badge/Paper-IEEE_Two--Column_(PDF%20%7C%20DOCX%20%7C%20TeX)-00629B?style=for-the-badge&logo=ieee)](./IEEE_Paper_Krishnendu_Arav.pdf)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](./simulate_and_optimize.py)
[![Validation: 100% Tests Passed](https://img.shields.io/badge/Tests-14%2F14%20Passed-10B981?style=for-the-badge)](./tests/test_scientific_validity.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](./LICENSE)

**Authors:**
- **Krishnendu** — *Department of Computer Science and Engineering, Manipal Institute of Technology (MIT), Manipal Academy of Higher Education (MAHE), Manipal, Karnataka 576104, India*
- **Arav Aggarwal** — *Department of Civil Engineering, School of Civil & Chemical Engineering, Manipal Institute of Technology (MIT), Manipal Academy of Higher Education (MAHE), Manipal, Karnataka 576104, India*

---

## Overview

This repository contains the complete, independently audited, and reproducible research codebase, datasets, and deliverables accompanying our IEEE paper. The study investigates a $500\text{ kg H}_2/\text{h}$ ($4,354.5\text{ t/yr}$) green hydrogen facility powered by hybrid wind and solar resources in Karnataka, India, supplying industrial off-takers in the Udupi–Mangalore coastal corridor (e.g., MRPL and MCF).

### Core Scientific Contributions & Audited Discoveries
1. **Multi-Year Meteorological Analysis (6 Contiguous Years, 52,608 Hours, 2019–2024):** Utilizes uninflated NASA POWER MERRA-2 reanalysis data. Proves that local coastal wind in Udupi–Mangalore ($13.35^\circ\text{N}, 74.79^\circ\text{E}$) yields a low capacity factor ($\overline{\text{CF}} = 14.70\%$), whereas open-access wheeled wind from the Chitradurga–Gadag highland corridor ($14.23^\circ\text{N}, 76.40^\circ\text{E}$, delivered $\overline{\text{CF}} = 43.29\%$) achieves strong seasonal complementarity with on-site coastal solar PV ($\overline{\text{CF}} = 21.23\%$).
2. **First-Principles High-Current PEMWE Calibration:** Reversible potential $V_{\text{rev}}(75^\circ\text{C}, 30\text{ bar}) = 1.234\text{ V}$ using Antoine saturation vapor pressure ($0.3846\text{ bar}$). Calibrated against $N=14$ experimental points (Nafion 115, $75^\circ\text{C}$, $30\text{ bar}$) from Bernt & Gasteiger (2016) and Rakousky et al. (2016), achieving $\text{RMSE} = 3.10\text{ mV}$ and $R^2 = 0.99971$. Active catalytic recombination ($\eta_{\text{recomb}} = 0.88$) and $i_{\min} = 0.20\text{ A/cm}^2$ guarantee anodic $\text{H}_2$-in-$\text{O}_2 \le 1.64\text{ mol\%} < 2.0\text{ mol\%}$ ($50\%$ LEL safety limit).
3. **Leak-Free Physics-Guided Surrogate (`Green-MLP`):** Strict $2 \times 5$ factorial ablation across 2 feature representations (Raw 5D vs.\ Leak-Free 12D Physics) and 5 model families. Green-MLP ($4,612$ parameters, $8,784$ FLOPs) achieves 5-fold CV $R^2 = 0.9943 \pm 0.0031$ ($R^2_{\text{LCOH}} = 0.9982$, $\text{RMSE} = ₹1.66 \pm 0.51/\text{kg}$), an $89.94\%$ FLOP reduction over regularized deep networks, and a $4,206\times$ inference acceleration over Numba-compiled 52,608-hour DAE simulations. Out-of-distribution Pareto test confirms rank preservation ($\rho = 0.9994$).
4. **Constrained NSGA-II & Split-CAPEX Dynamics:** Sizing power electronics and balance-of-plant (BOP) for peak megawatt overload throughput ($P_{\text{PEM,peak}}$) adds more capital cost than is saved on stack cell area. The optimal knee point operates at $i_{\max}^* = 1.764\text{ A/cm}^2$, achieving $\text{LCOH} = ₹332.51/\text{kg H}_2$ (\$$3.96/\text{kg}$), $\text{LPSP} = 0.62\%$, and $12.0$-year stack life. Wheeled highland wind lowers LCOH by $12.96\%$ relative to pure coastal generation ($₹382.04/\text{kg}$) and prevents catastrophic supply failure ($\text{LPSP} = 21.30\%$) under coastal wind droughts.
5. **Multivariate Statistical Process Monitoring Reality:** Latent-variable PCA ($k=3$, $96.03\%$ variance) across 9 measurable industrial sensors does not detect incipient faults earlier than a well-designed load-compensated 1D voltage-residual EWMA chart ($11.6\text{ h}$ vs.\ $10.6\text{ h}$ latency across 100 Monte Carlo runs). However, PCA provides indispensable multi-sensor root-cause diagnosis via contribution plots ($95.2\%$ thermal-pressure isolation on dehydration, $100\%$ detection on anode flooding).

---

## Repository Structure

```text
├── IEEE_Paper_Krishnendu_Arav.pdf       # Compiled Two-Column IEEE Conference Paper (PDF)
├── IEEE_Paper_Krishnendu_Arav.docx      # Editable Two-Column IEEE Manuscript (DOCX)
├── IEEE_Paper_Krishnendu_Arav.tex       # Official IEEEtran LaTeX Source Code
├── IEEE_Paper_Krishnendu_Arav.html      # High-Resolution KaTeX HTML Source
├── simulate_and_optimize.py             # Master simulation, surrogate, NSGA-II & PCA pipeline
├── generate_ieee_pdf_and_docx.py        # PDF & DOCX automated builder
├── requirements.txt                     # Python package dependencies
├── VALIDATION_REPORT.md                 # Experimental & physical validation report
├── RESEARCH_AUDIT_REPORT.md             # 5-specialist critical audit report
├── REPRODUCIBILITY_REPORT.md            # Exact execution environment & reproduction protocol
├── PUBLICATION_STRATEGY.md              # Target venues, reviewer defense & publication roadmap
├── data/
│   ├── karnataka_multiyear_meteo_2019_2024.csv # 52,608-hour uninflated NASA POWER data
│   ├── pem_polarization_calibration_exp.csv    # Experimental polarization benchmark (14 points)
│   └── coastal_karnataka_meteo_hourly.csv      # 1-year legacy dataset reference
├── results/
│   ├── summary_metrics.json             # Complete verified metrics (759 lines)
│   ├── pareto_front.csv                 # Exact-verified NSGA-II Pareto solutions
│   └── surrogate_ablation_2x5.csv       # 5-fold CV metrics for 2x5 factorial experiment
├── tests/
│   └── test_scientific_validity.py      # Automated unit test suite (14 tests)
└── figures/
    ├── fig1_coastal_meteo_and_pem_polarization.png
    ├── fig2_green_surrogate_parity_and_efficiency.png
    ├── fig3_pareto_front_3d_tradeoffs.png
    ├── fig4_xai_feature_sensitivity_and_uncertainty.png
    ├── fig5_pca_fault_monitoring_t2_spe.png
    └── fig6_dynamic_dispatch_monsoon_vs_summer.png
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Krishnendu409/CHEMIGNITE-Green-H2-Research.git
cd CHEMIGNITE-Green-H2-Research

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Master Simulation Pipeline
Executes Numba-accelerated 52,608-hour multi-year simulation, surrogate training, 5-seed constrained NSGA-II, true $N=5,000$ Monte Carlo UQ, and 100-realization PCA fault monitoring (~156 seconds total runtime):
```bash
python simulate_and_optimize.py
```

### 3. Run Automated Validation Test Suite
```bash
pytest tests/test_scientific_validity.py -v
```
All 14 unit tests verify mathematical, physical, and economic invariants:
- Test 1: Multi-year dataset completeness and physical bounds (52,608 hours, 2019–2024)
- Test 2: Nernst reversible potential exactness at $75^\circ\text{C}, 30\text{ bar}$ ($1.234\text{ V} \pm 1\text{ mV}$)
- Test 3: Semi-empirical polarization experimental calibration ($\text{RMSE} < 10\text{ mV}$, $R^2 > 0.999$)
- Test 4: Crossover safety under turndown ($y_{\text{H}_2} < 2.0\text{ mol\%}$)
- Test 5: Dynamic voltage degradation non-negativity
- Test 6: Surrogate feature leak-free formulation (zero economic weights)
- Test 7: Surrogate 5-fold CV accuracy ($R^2 > 0.99$, $\text{RMSE} < 3.0\text{ INR/kg}$)
- Test 8: Deb's feasibility-first constraint handling
- Test 9: Sizing baselines reliability compliance ($\text{LPSP} \le 1.0\%$)
- Test 10: Split-CAPEX electrolyzer cost scaling
- Test 11: End-to-end numerical consistency with `summary_metrics.json`
- Test 12: Pareto front feasibility and 9-sensor PCA integrity (100 MC runs, no unmeasurable $\sigma_{\text{mem}}$)
- Test 13: Polarization calibration train-validation partition ($N_{\text{cal}}=10, N_{\text{val}}=4$, $\text{RMSE} < 4.0\text{ mV}$)
- Test 14: PCA autocorrelation isolation ($\rho_1(T^2) > 0.65$), $Q$-alone $\text{FAR} < 3\%$, and wheeling breakeven bounds

### 4. Build IEEE Paper Documents (PDF & DOCX)
```bash
python generate_ieee_pdf_and_docx.py
```

---

## Key Quantitative Benchmark Summary

| Metric / Parameter | Audited Value | Benchmark / Baseline | Significance |
| :--- | :---: | :---: | :--- |
| **Experimental Calibration Fit** | $\text{RMSE} = 3.10\text{ mV}, R^2 = 0.99971$ | $N=14$ Exp.\ Points (Bernt 2016) | Exact validation up to $3.0\text{ A/cm}^2$ |
| **Reversible Nernst Potential** | $1.234\text{ V}$ | Antoine $P_{\text{sat}} = 0.3846\text{ bar}$ | Corrects historical Tetens equation bug |
| **Green-MLP Parameters & FLOPs** | $4,612$ params, $8,784$ FLOPs | Regularized-DNN: $94,464$ FLOPs | **$-89.94\%$ FLOPs**, $4,206\times$ speedup |
| **Surrogate Out-of-Sample $R^2$** | $0.9943 \pm 0.0031$ | GBDT: $0.9213$, GP: $0.9888$ | High accuracy without target leakage |
| **Pareto Spearman Rank Correlation** | $\rho_{\text{LCOH}} = 0.9994, \rho_{\text{deg}} = 0.9960$ | Held-out Pareto Set ($52,608\text{ h}$) | Perfect evolutionary rank preservation |
| **Optimal Knee-Point LCOH** | **₹332.51/kg ($3.96/kg)** | Fixed $i_{\text{nom}}$ ($1.80$): ₹329.12/kg | Split BOP CAPEX offsets overload area savings |
| **Siting Advantage (Regime B vs A)** | **₹332.51/kg vs ₹382.04/kg** | Wheeled Wind vs Coastal Hybrid | **$-12.96\%$ LCOH** via highland wind |
| **Coastal Drought Failure** | $\text{LPSP} = 21.30\%$ | Knee Hardware in Coastal Regime | Proves necessity of regional grid wheeling |
| **Single-Resource Sizing Baselines** | Wind-Only: ₹550.69, Solar-Only: ₹449.27 | Reliability-Matched ($\text{LPSP} \le 1\%$) | Hybrid plant is $26.0\%\text{--}39.6\%$ cheaper |
| **5,000-Sample Monte Carlo Mean** | $₹335.33 \pm 24.33/\text{kg}$ | 90% CI: $[₹296.58, ₹376.66]$ | $73.3\%$ probability $< ₹350/\text{kg}$ |
| **MSPM Dehydration Detection** | Latency: $11.6 \pm 3.2\text{ h}$, $\text{FAR} = 9.80\%$ | 1D EWMA: Latency $10.6\text{ h}$, $\text{FAR} = 5.39\%$ | PCA provides multi-sensor root cause ($95.2\%$) |

---

## Citation & Contact

If you use this codebase or methodology in your research, please cite:

```bibtex
@inproceedings{krishnendu2026physics,
  title={Physics-Guided Surrogate Optimization and Load-Compensated Multivariate Fault Monitoring of a Coastal Hybrid Wind--Solar--PEM Green Hydrogen Plant under Multi-Year Meteorological Uncertainty},
  author={Krishnendu and Aggarwal, Arav},
  booktitle={Proc. 2026 National Conference on Artificial Intelligence, Sustainable Processes, Innovation & Renewable Energy (CHEMIGNITE / ASPIRE 2026)},
  address={Manipal, Karnataka, India},
  year={2026}
}
```
