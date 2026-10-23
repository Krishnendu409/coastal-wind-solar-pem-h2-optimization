import subprocess
import pathlib
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION

BASE = pathlib.Path(r"c:\Users\krish\Desktop\CHEMIGNITE\01_RESEARCH_PAPER_WIND_PEM_H2")
PACK = pathlib.Path(r"c:\Users\krish\Desktop\CHEMIGNITE\00_OCT25_SUBMISSION_PACK")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# -------------------------------------------------------------
# 1. Compile IEEE HTML to High-Res PDF via Headless Chrome
# -------------------------------------------------------------
html_path = BASE / "IEEE_Paper_Krishnendu_Arav.html"
pdf_path = BASE / "IEEE_Paper_Krishnendu_Arav.pdf"

print("Compiling IEEE Two-Column PDF via Headless Chrome...")
subprocess.run([
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    "--virtual-time-budget=6000",
    html_path.as_uri()
], check=True)
print(f"Compiled IEEE Two-Column PDF: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
shutil.copy2(pdf_path, PACK / "IEEE_Paper_Krishnendu_Arav.pdf")

# -------------------------------------------------------------
# 2. Build Two-Column IEEE Word (.docx) with python-docx
# -------------------------------------------------------------
print("Building Two-Column IEEE Word Document (.docx)...")
doc = docx.Document()

# Page setup: US Letter, IEEE Margins (Top 0.75", Bottom 1.0", Left/Right 0.625")
sec1 = doc.sections[0]
sec1.page_width = Inches(8.5)
sec1.page_height = Inches(11.0)
sec1.top_margin = Inches(0.75)
sec1.bottom_margin = Inches(1.0)
sec1.left_margin = Inches(0.625)
sec1.right_margin = Inches(0.625)

style_normal = doc.styles['Normal']
style_normal.font.name = 'Times New Roman'
style_normal.font.size = Pt(10)
style_normal.paragraph_format.line_spacing = 1.15
style_normal.paragraph_format.space_after = Pt(0)
style_normal.paragraph_format.space_before = Pt(0)

# Conference Header
p_banner = doc.add_paragraph()
p_banner.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_ban = p_banner.add_run("2026 NATIONAL CONFERENCE ON ARTIFICIAL INTELLIGENCE, SUSTAINABLE PROCESSES, INNOVATION & RENEWABLE ENERGY (CHEMIGNITE / ASPIRE 2026) • MIT MANIPAL")
r_ban.font.name = 'Times New Roman'
r_ban.font.size = Pt(7.5)
p_banner.paragraph_format.space_after = Pt(8)

# Title (20 pt Times New Roman Centered)
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_title = p_title.add_run("Physics-Guided Surrogate Optimization and Load-Compensated Multivariate Fault Monitoring of a Coastal Hybrid Wind–Solar–PEM Green Hydrogen Plant under Multi-Year Meteorological Uncertainty")
r_title.font.name = 'Times New Roman'
r_title.font.size = Pt(20)
p_title.paragraph_format.space_after = Pt(12)

# Author Block (2-column invisible table)
auth_table = doc.add_table(rows=1, cols=2)
auth_table.alignment = WD_TABLE_ALIGNMENT.CENTER
auth_cells = auth_table.rows[0].cells

def fill_author(cell, name, dept, inst, uni, loc):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r1 = p.add_run(name + "\n")
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(11)
    for line in [dept, inst, uni]:
        r = p.add_run(line + "\n")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)
        r.italic = True
    r_loc = p.add_run(loc)
    r_loc.font.name = 'Times New Roman'
    r_loc.font.size = Pt(9.5)

fill_author(
    auth_cells[0],
    "Krishnendu",
    "Department of Computer Science and Engineering",
    "Manipal Institute of Technology (MIT)",
    "Manipal Academy of Higher Education (MAHE)",
    "Manipal, Karnataka 576104, India"
)
fill_author(
    auth_cells[1],
    "Arav Aggarwal",
    "Department of Civil Engineering, School of Civil & Chemical Engineering",
    "Manipal Institute of Technology (MIT)",
    "Manipal Academy of Higher Education (MAHE)",
    "Manipal, Karnataka 576104, India"
)

# 2-Column Section Break
sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
sec2.top_margin = Inches(0.75)
sec2.bottom_margin = Inches(1.0)
sec2.left_margin = Inches(0.625)
sec2.right_margin = Inches(0.625)
sectPr = sec2._sectPr
cols = sectPr.xpath('./w:cols')[0]
cols.set(qn('w:num'), '2')
cols.set(qn('w:space'), '360') # 360 twips = 0.25 inch

# Abstract (9pt Bold)
p_abs = doc.add_paragraph()
p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_abs.paragraph_format.first_line_indent = Inches(0.15)
p_abs.paragraph_format.space_after = Pt(6)
r_ab1 = p_abs.add_run("Abstract")
r_ab1.font.size = Pt(9)
r_ab1.bold = True
r_ab1.italic = True
r_ab2 = p_abs.add_run(
    "—Decarbonizing heavy coastal industries—such as petroleum refining, nitrogenous fertilizers, and port logistics in southwestern "
    "India—requires continuous, cost-effective green hydrogen (H₂) supply under severe seasonal weather variability. In this work, we "
    "investigate a 500 kg H₂/h (4,355 t/yr) hybrid wind–solar proton exchange membrane water electrolysis (PEMWE) facility through an "
    "audited, multi-year Process Systems Engineering (PSE) and physics-guided machine learning framework. First, analysis of six contiguous "
    "years (2019–2024, 52,608 hourly observations) of uninflated NASA POWER MERRA-2 meteorological data demonstrates that local coastal wind "
    "in the Udupi–Mangalore corridor (13.35°N, 74.79°E) yields a low capacity factor (CF = 14.70%), whereas open-access wheeled wind from the "
    "Chitradurga–Gadag highland corridor (14.23°N, 76.40°E, delivered CF = 43.29%) achieves strong seasonal complementarity with on-site coastal "
    "solar PV (CF = 21.23%). Second, a first-principles high-current PEMWE model calibrated against published experimental data (14 points, "
    "Nafion 115, 75°C, 30 bar, RMSE = 3.10 mV, R² = 0.99971) is coupled with catalytic recombination to guarantee anodic H₂-in-O₂ ≤ 1.64 mol% "
    "(<50% of lower explosion limit) at turndown (i_min = 0.20 A/cm²). Third, a strictly leak-free 2×5 surrogate ablation establishes that a compact "
    "physics-guided Multi-Layer Perceptron (Green-MLP, 4,612 parameters) achieves 5-fold cross-validation R² = 0.9943 ± 0.0031 (R² = 0.9982 on "
    "LCOH, RMSE = INR 1.66 ± 0.51/kg) with an 89.94% FLOP reduction over regularized deep networks and a 4,206× inference acceleration over "
    "Numba-accelerated 52,608-hour simulations. Fourth, a 5-seed constrained NSGA-II optimization with split-CAPEX modeling reveals a critical "
    "trade-off: dynamic current overload (i_max > 1.80 A/cm²) does not decrease Levelized Cost of Hydrogen (LCOH) because balance-of-plant (BOP) "
    "and rectifier capital costs scale with peak megawatt capacity, favoring a knee-point design (48.02 MW wind, 72.77 MW solar, 43.61 MW nominal "
    "PEM at i_max* = 1.764 A/cm², 86.86 MWh battery) delivering LCOH = INR 332.51/kg H₂ ($3.96/kg), LPSP = 0.62%, and 12.0-year stack lifetime. "
    "Wheeled highland wind lowers LCOH by 12.96% relative to an optimized pure coastal hybrid plant (INR 382.04/kg) and avoids severe supply failure "
    "(LPSP = 21.30%) if identical hardware operates solely on coastal wind. Finally, a 9-sensor latent-variable Principal Component Analysis (PCA, "
    "k=3, 96.03% variance) benchmarked across 100 dynamic realizations demonstrates that while load-compensated 1D voltage-residual EWMA detects "
    "incipient membrane dehydration with marginally lower latency (10.6 h vs. 11.6 h), PCA provides unique multi-sensor root-cause diagnosis via "
    "contribution plots, isolating thermal-pressure anomalies (95.2% combined contribution) during dehydration and overpotential-production anomalies "
    "during anodic flooding (100% detection rate)."
)
r_ab2.font.size = Pt(9)
r_ab2.bold = True

# Index Terms
p_kw = doc.add_paragraph()
p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_kw.paragraph_format.first_line_indent = Inches(0.15)
p_kw.paragraph_format.space_after = Pt(8)
r_kw1 = p_kw.add_run("Index Terms")
r_kw1.font.size = Pt(9)
r_kw1.bold = True
r_kw1.italic = True
r_kw2 = p_kw.add_run(
    "—Explainable AI (XAI), green deep learning, green hydrogen, multi-objective optimization (NSGA-II), multi-year meteorological "
    "uncertainty, multivariate statistical process monitoring (PCA), PEM water electrolysis, techno-economic analysis."
)
r_kw2.font.size = Pt(9)
r_kw2.bold = True

def add_sec_heading(title_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title_text.upper())
    r.font.size = Pt(10)

def add_subsec_heading(title_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title_text)
    r.font.size = Pt(10)
    r.italic = True

def add_body_p(text, indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.15)
    r = p.add_run(text)
    r.font.size = Pt(10)
    return p

def add_eq(eq_text, eq_num):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"{eq_text}    ({eq_num})")
    r.font.size = Pt(9.5)
    r.italic = True

def add_fig(img_file, cap_text):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(str(BASE / "figures" / img_file), width=Inches(3.35))
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap.paragraph_format.space_after = Pt(6)
    r_cap = p_cap.add_run(cap_text)
    r_cap.font.size = Pt(8)

# Section I
add_sec_heading("I. Introduction")
add_body_p(
    "THE National Green Hydrogen Mission of India targets 5 MMT/yr of green hydrogen production capacity by 2030 to substitute "
    "fossil-derived grey hydrogen across petroleum refining, ammonia-based nitrogenous fertilizers, and steelmaking [1]. Heavy "
    "chemical complexes in coastal Karnataka—such as Mangalore Refinery and Petrochemicals Limited (MRPL) and Mangalore Chemicals "
    "& Fertilizers (MCF)—represent substantial concentrated hydrogen demands (>100 t/day). Siting green hydrogen facilities within coastal "
    "industrial corridors offers immediate proximity to off-takers, port bunkering terminals, and abundant seawater desalination potential [2].",
    indent=False
)
add_body_p(
    "However, powering large-scale water electrolysis exclusively from on-site variable renewable energy (VRE) encounters severe "
    "techno-economic hurdles in tropical coastal regions. Tropical monsoon climates experience extreme seasonal weather shifts: solar "
    "irradiance drops precipitously during the four-month Southwest Monsoon (June–September) under dense cloud cover, whereas coastal "
    "surface winds often remain inadequate to sustain baseload multi-megawatt electrolysis throughout the remainder of the year [3], [4]."
)
add_body_p(
    "Recent chemical engineering and energy systems literature has focused on three core engineering strategies: "
    "(1) High-Current-Density PEM Electrolysis (Masampally, Mitra, and Runkana [5]), where operating at i ∈ [2.0, 3.5] A/cm² slashes "
    "stack CAPEX per unit throughput at the expense of two-phase porous transport layer (PTL) bubble crowding and dynamic voltage degradation [6]–[8]; "
    "(2) Physics-Informed 'Green Deep Learning' Surrogates (Pujari, Inapakurthi, and Mitra [2]), replacing over-parameterized black-box "
    "deep neural networks with compact, physics-guided neural architectures [9]–[12]; and (3) Multivariate Statistical Process Monitoring "
    "(MSPM) via Principal Component Analysis (PCA) Hotelling's T² and Jackson–Mudholkar Q/SPE charts (Madakyaru et al. [13]–[16]) to isolate "
    "incipient membrane dehydration and anodic flooding under dynamic renewable ramping. This paper bridges this gap through audited multi-year simulations."
)

# Section II
add_sec_heading("II. Regional Meteorology, Siting, and Calibrated First-Principles Electrochemistry")
add_subsec_heading("A. Multi-Year Meteorological Analysis and Regional Siting")
add_body_p(
    "Hourly meteorological data spanning six complete years (2019–2024, N = 52,608 h) were extracted from the NASA POWER MERRA-2 "
    "archive [17]. We examine two distinct geographic corridors within Karnataka, India: (1) On-Site Coastal Corridor (13.3525°N, 74.7928°E) "
    "with GHI = 213.92 W/m², tilted POA irradiance = 220.59 W/m² (PV CF = 21.23%), and raw 100 m wind speed = 4.15 m/s (wind CF = 14.70%); "
    "and (2) Inland Highland Wind Corridor (Chitradurga–Gadag, 14.225°N, 76.398°E) with 100 m wind speed = 6.69 m/s (delivered CF = 43.29% "
    "after 3.4% transmission losses). Under KERC regulations [18], wheeling charges are INR 0.58/kWh.",
    indent=False
)
add_fig(
    "fig1_coastal_meteo_and_pem_polarization.png",
    "Fig. 1. (a) Seasonal complementarity across 6 years (2019–2024, 52,608 h) between coastal solar PV and open-access wheeled highland wind; (b) Calibrated PEMWE polarization curve against experimental data (75°C, 30 bar) with overpotential breakdown."
)

add_subsec_heading("B. First-Principles High-Current PEMWE Formulation")
add_body_p(
    "Single-cell potential V_cell(i, T, P) over i ∈ [0.05, 3.50] A/cm² at T = 348.15 K (75°C) and P_cat = 30 bar is decomposed into "
    "reversible Nernst potential (V_rev = 1.234 V using Antoine vapor pressure 0.3846 bar), Butler–Volmer activation (η_act), "
    "Nafion 115 ohmic overpotential (η_ohm, δ_mem = 127 μm, λ = 18.5), and two-phase bubble crowding concentration overpotential (η_conc):",
    indent=False
)
add_eq("V_cell = V_rev(T, P) + η_act(i, T) + η_ohm(i, T, λ) + η_conc(i, T)", "1")
add_eq("η_conc = -(RT/4F)ln(1 - i/i_L,an) - (RT/2F)ln(1 - i/i_L,cat) + β_bub · i^1.65", "2")
add_eq("dV_deg/dt = 1.20 + 0.25 + 2.80 · max(0, i(t) - 1.80)^1.85 + 1.40 · |di/dt|   [μV/h]", "3")
add_body_p(
    "Model parameters were calibrated against N = 14 experimental data points from Bernt & Gasteiger [6] and Rakousky et al. [7] (75°C, 30 bar), "
    "achieving RMSE = 3.10 mV, MAE = 3.01 mV, and R² = 0.99971. Catalytic recombination (η_recomb = 0.88) and i_min = 0.20 A/cm² limit anodic "
    "H₂-in-O₂ to 1.64 mol% (<50% LEL trip limit)."
)

# Table 1 in docx
t1 = doc.add_table(rows=7, cols=8)
t1.alignment = WD_TABLE_ALIGNMENT.CENTER
t1_headers = ["i (A/cm²)", "V_rev (V)", "η_act (V)", "η_ohm (V)", "η_conc (V)", "V_cell (V)", "SEC (kWh/kg)", "y_H2 (%)"]
for j, h in enumerate(t1_headers):
    cell = t1.cell(0, j)
    cell.paragraphs[0].text = h
    cell.paragraphs[0].runs[0].font.size = Pt(7.5)
    cell.paragraphs[0].runs[0].font.bold = True
t1_data = [
    ["0.50", "1.234", "0.293", "0.056", "0.007", "1.590", "46.69", "0.663"],
    ["1.00", "1.234", "0.326", "0.113", "0.019", "1.691", "48.86", "0.332"],
    ["1.80", "1.234", "0.355", "0.203", "0.046", "1.838", "52.62", "0.185"],
    ["2.50", "1.234", "0.371", "0.281", "0.078", "1.965", "56.01", "0.133"],
    ["3.00", "1.234", "0.380", "0.338", "0.106", "2.058", "58.51", "0.111"],
    ["3.50", "1.234", "0.388", "0.394", "0.137", "2.153", "61.11", "0.095"],
]
for i, row in enumerate(t1_data):
    for j, val in enumerate(row):
        cell = t1.cell(i+1, j)
        cell.paragraphs[0].text = val
        cell.paragraphs[0].runs[0].font.size = Pt(7.2)

# Section III
add_sec_heading("III. Physics-Guided Surrogate Architecture and Controlled Ablation")
add_body_p(
    "To enable rapid evolutionary search across 52,608 hours, we formulate a strictly leak-free 12D dimensionless feature map Φ(x) "
    "incorporating physical capacity ratios, voltage overpotentials, and Faradaic efficiency with zero target cost leakage. A 2×5 controlled "
    "ablation benchmark (Latin Hypercube N = 360) compares Raw 5D vs. Physics 12D representations across five model families under 5-fold cross-validation.",
    indent=False
)
add_fig(
    "fig2_green_surrogate_parity_and_efficiency.png",
    "Fig. 2. Physics-guided surrogate validation: (a) Parity plot for LCOH (R² = 0.9982, RMSE = INR 1.66/kg); (b) Degradation parity (R² = 0.9946); (c) Compute savings showing an 89.94% FLOP reduction and 4,206× inference speedup."
)
add_body_p(
    "The proposed Green-MLP (4,612 parameters, 8,784 FLOPs) achieves 5-fold CV R² = 0.9943 ± 0.0031 (LCOH RMSE = INR 1.66 ± 0.51/kg), "
    "slashing forward FLOPs by 89.94% relative to regularized deep networks (94,464 FLOPs) while outperforming them in accuracy. "
    "Evaluating Green-MLP on the Pareto set confirms near-perfect rank preservation: Spearman ρ = 0.9994 on LCOH and ρ = 0.9960 on degradation."
)

# Section IV
add_sec_heading("IV. Multi-Objective Optimization, Siting Regimes, and Uncertainty Quantification")
add_body_p(
    "Using 5-seed constrained NSGA-II [20] with Deb's feasibility-first tournament selection (LPSP ≤ 1.0%, StackLife ≥ 7.5 yr), we minimize "
    "LCOH, LPSP, and Lifecycle Carbon Intensity + Degradation under Split-CAPEX modeling. Stack CAPEX (INR 3.364 Cr/MW) scales with nominal capacity, "
    "while power electronics and BOP (INR 2.436 Cr/MW) scale with peak overload megawatt capacity P_PEM,peak.",
    indent=False
)
add_fig(
    "fig3_pareto_front_3d_tradeoffs.png",
    "Fig. 3. (a) 3D Pareto frontier evaluated across 6 weather years; (b) 2D projection highlighting Knee-Point Optimal Design against single-resource and sizing baselines."
)
add_body_p(
    "Addressing Hypothesis H1: The optimal knee point operates at i_max* = 1.764 A/cm² (below nominal 1.80 A/cm²). Sizing at fixed nominal current "
    "(i_max = 1.80 A/cm²) achieves INR 329.12/kg (1.03% lower). Pushing current density above 2.0 A/cm² does not reduce LCOH because BOP CAPEX "
    "scaling outweighs membrane area savings. Siting in highland wind Regime B (INR 332.51/kg) lowers LCOH by 12.96% relative to coastal Regime A "
    "(INR 382.04/kg) and prevents catastrophic supply failure (LPSP = 21.30%) if operated solely on coastal wind."
)
add_fig(
    "fig4_xai_feature_sensitivity_and_uncertainty.png",
    "Fig. 4. (a) Permutation feature importance across objectives; (b) 5,000-sample Monte Carlo LCOH distribution (P05 = INR 296.58/kg, Mean = INR 335.33/kg, P95 = INR 376.66/kg)."
)
add_body_p(
    "True physics-evaluated Monte Carlo simulation (N = 5,000) yields mean LCOH = INR 335.33 ± 24.33/kg with a 90% confidence interval of "
    "[INR 296.58, INR 376.66]/kg (73.3% probability < INR 350/kg). Sobol sensitivity shows Renewable CAPEX (56.65%), WACC (21.58%), and PEM CAPEX "
    "(18.89%) dominate output uncertainty, while weather resource volatility contributes <1.3%."
)

# Section V
add_sec_heading("V. Load-Compensated Multivariate Statistical Process Monitoring (MSPM)")
add_body_p(
    "Under dynamic renewable dispatch (i ∈ [0.20, 1.76] A/cm²), 9 physically measurable industrial sensors are monitored via PCA (k = 3 PCs, "
    "96.03% variance). PCA was benchmarked against a Load-Compensated 1D Voltage-Residual EWMA filter across 100 independent Monte Carlo realizations.",
    indent=False
)
add_fig(
    "fig5_pca_fault_monitoring_t2_spe.png",
    "Fig. 5. Multivariate statistical fault monitoring: (a) Hotelling's T² control chart; (b) Jackson–Mudholkar Q/SPE chart; (c) Sensor contribution diagnosis isolating thermal-pressure anomalies during Fault 1 and voltage-pressure anomalies during Fault 2."
)
add_body_p(
    "Addressing Hypothesis H3: PCA does not detect incipient faults faster than load-compensated EWMA (11.6 h latency for PCA vs. 10.6 h for EWMA; "
    "FAR = 9.80% vs. 5.39%). However, PCA provides indispensable multi-sensor root-cause diagnosis via contribution plots, isolating Stack Temperature "
    "(60.17%) and Cathode Pressure (35.07%) during membrane dehydration, and Cell Voltage (34.61%) and Cathode Pressure (26.04%) during anodic flooding."
)

# Section VI
add_sec_heading("VI. Seasonal Dynamic Dispatch Verification")
add_body_p(
    "Fig. 6 verifies hourly dynamic operation across a summer week (solar-dominant diurnal high-current ramping up to 1,050 kg/h with buffer storage) "
    "and a monsoon week (steady highland wind baseload electrolysis at 480–540 kg/h with minimal battery cycling), confirming uninterrupted supply.",
    indent=False
)
add_fig(
    "fig6_dynamic_dispatch_monsoon_vs_summer.png",
    "Fig. 6. Hourly dispatch profiles of the Knee-Point Plant (168 h): (a) Pre-monsoon summer week (solar diurnal ramping); (b) Southwest Monsoon week (steady baseload wind); (c) Compressed H₂ buffer storage and battery state of charge."
)

# Section VII & VIII
add_sec_heading("VII. Conclusion")
add_body_p(
    "This study established an audited Process Systems Engineering and Green AI blueprint for an industrial coastal green hydrogen plant in "
    "southwestern India. Key findings include: (1) Split-CAPEX dynamics favor nominal operating densities (i_max* = 1.764 A/cm², LCOH = INR 332.51/kg, "
    "LPSP = 0.62%); (2) Wheeled highland wind lowers LCOH by 12.96% and prevents severe coastal wind drought failure; (3) Leak-free Green-MLP achieves "
    "R² = 0.9943 with an 89.94% FLOP reduction and 4,206× speedup; and (4) PCA provides critical multi-sensor root-cause isolation even when load-compensated "
    "univariate filters detect marginally earlier.",
    indent=False
)

add_sec_heading("References")
refs = [
    "[1] Ministry of New and Renewable Energy (MNRE), National Green Hydrogen Mission. New Delhi, India: Government of India, Jan. 2023.",
    "[2] N. K. Pujari, R. K. Inapakurthi, and K. Mitra, 'Integrated forecasting and design of distributed wind-to-hydrogen systems for urban sustainability,' Comput. Chem. Eng., vol. 211, p. 108842, Aug. 2026. DOI: 10.1016/j.compchemeng.2026.108842.",
    "[3] K. Mitra, R. Everson, and J. Fieldsend, Eds., Optimization, Uncertainty and Machine Learning in Wind Energy Conversion Systems. Cham: Springer Nature, Jan. 2025. DOI: 10.1007/978-3-031-72124-3.",
    "[4] Abhishek, N. K. Pujari, and K. Mitra, 'AI-driven wind resource modelling for efficient wind-to-hydrogen system design at Jaisalmer,' in Proc. Int. Conf. AI Manuf. Sustain. (AIMS), Dec. 2025, pp. 1–8.",
    "[5] V. S. Masampally, K. Mitra, and V. Runkana, 'Modeling and simulation of hydrogen production in a PEM water electrolyzer at high current densities,' Energy Convers. Manage., vol. 352, p. 121110, 2026. DOI: 10.1016/j.enconman.2026.121110.",
    "[6] M. Bernt and H. A. Gasteiger, 'Influence of the porous transport layer on PEM water electrolysis performance at high current densities,' J. Electrochem. Soc., vol. 163, no. 11, pp. F3179–F3189, Sep. 2016. DOI: 10.1149/2.0651611jes.",
    "[7] C. Rakousky et al., 'An analysis of degradation phenomena in polymer electrolyte membrane water electrolysis,' J. Power Sources, vol. 326, pp. 120–128, Sep. 2016. DOI: 10.1016/j.jpowsour.2016.06.082.",
    "[8] M. Chandesris et al., 'Membrane degradation in PEM water electrolyzer: Numerical modeling and experimental validation,' Int. J. Hydrogen Energy, vol. 40, no. 3, pp. 1353–1366, Jan. 2015. DOI: 10.1016/j.ijhydene.2014.11.111.",
    "[9] P. Trinke, P. Bensmann, and R. Hanke-Rauschenbach, 'Current density effect on hydrogen crossover in PEM water electrolysis,' J. Electrochem. Soc., vol. 165, no. 7, pp. F502–F513, May 2018. DOI: 10.1149/2.0461807jes.",
    "[10] R. K. Inapakurthi, N. K. Pujari, and K. Mitra, 'Maximizing energy production from wind farm layout using model predictive control,' in Optimization, Uncertainty and Machine Learning in Wind Energy Conversion Systems. Cham: Springer, 2025, pp. 85–112.",
    "[11] R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni, 'Green AI,' Commun. ACM, vol. 63, no. 12, pp. 54–63, Dec. 2020. DOI: 10.1145/3381831.",
    "[12] M. Raissi, P. Perdikaris, and G. E. Karniadakis, 'Physics-informed neural networks,' J. Comput. Phys., vol. 378, pp. 686–707, Feb. 2019. DOI: 10.1016/j.jcp.2018.10.045.",
    "[13] S. J. Qin, 'Survey on data-driven industrial process monitoring and diagnosis,' Annu. Rev. Control, vol. 36, no. 2, pp. 220–234, Dec. 2012. DOI: 10.1016/j.arcontrol.2012.09.004.",
    "[14] J. E. Jackson and G. S. Mudholkar, 'Control procedures for residuals associated with principal component analysis,' Technometrics, vol. 21, no. 3, pp. 341–349, Aug. 1979. DOI: 10.1080/00401706.1979.10489779.",
    "[15] M. Madakyaru, C. N. Bhushan, and S. R. Kulkarni, 'Multivariate statistical process monitoring and improved latent-variable fault detection for nonlinear chemical processes,' J. Process Control, vol. 134, p. 103148, 2024. DOI: 10.1016/j.jprocont.2024.103148.",
    "[16] M. N. Nounou, B. R. Bakshi, P. K. Goel, and X. Shen, 'ches-PCA and multiscale latent variable modeling for fault detection in dynamic chemical systems,' AIChE J., vol. 48, no. 8, pp. 1775–1793, 2002. DOI: 10.1002/aic.690480816.",
    "[17] NASA POWER Project, 'Prediction of Worldwide Energy Resources (Hourly Agroclimatology & Solar-Wind Energy Archive, Karnataka Regional Corridors),' NASA, 2026.",
    "[18] Karnataka Electricity Regulatory Commission (KERC), 'Terms and Conditions for Open Access and Renewable Wheeling Charges: Multi-Year Tariff Order,' Bengaluru, India, 2024.",
    "[19] T. E. Springer, T. A. Zawodzinski, and S. Gottesfeld, 'Polymer electrolyte fuel cell model,' J. Electrochem. Soc., vol. 138, no. 8, pp. 2334–2342, Aug. 1991. DOI: 10.1149/1.2085971.",
    "[20] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, 'A fast and elitist multiobjective genetic algorithm: NSGA-II,' IEEE Trans. Evol. Comput., vol. 6, no. 2, pp. 182–197, Apr. 2002. DOI: 10.1109/4235.996017.",
    "[21] Central Electricity Authority (CEA), 'CO2 Baseline Database for the Indian Power Sector (User Guide Version 19.0),' Ministry of Power, New Delhi, India, 2024."
]
for ref in refs:
    p_r = doc.add_paragraph()
    p_r.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_r.paragraph_format.space_after = Pt(2)
    rr = p_r.add_run(ref)
    rr.font.size = Pt(8)

docx_path = BASE / "IEEE_Paper_Krishnendu_Arav.docx"
doc.save(str(docx_path))
shutil.copy2(docx_path, PACK / "IEEE_Paper_Krishnendu_Arav.docx")
print(f"Saved Editable Two-Column IEEE Word Docx: {docx_path} ({docx_path.stat().st_size:,} bytes)")
print("Deliverables successfully updated and synchronized!")
