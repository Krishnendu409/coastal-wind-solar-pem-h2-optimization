import os
import pathlib
import re

BASE = pathlib.Path(r"c:\Users\krish\Desktop\CHEMIGNITE\01_RESEARCH_PAPER_WIND_PEM_H2")
html_path = BASE / "IEEE_Paper_Krishnendu_Arav.html"
tex_path = BASE / "IEEE_Paper_Krishnendu_Arav.tex"

print(f"Enhancing {html_path.name}...")
html_text = html_path.read_text(encoding="utf-8")

# 1. Update CSS in HTML
old_fig_css = """  /* Figures (Caption Below, 8pt) */
  .ieee-fig {
    margin: 7px 0;
    break-inside: avoid;
    text-align: center;
  }
  .ieee-fig img {
    width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 3px auto;
    border: 0.3pt solid #ddd;
  }
  .ieee-fig-cap {
    font-size: 8pt;
    line-height: 1.15;
    text-align: justify;
  }"""

new_fig_css = """  /* Figures (Full Width Across Both Columns, High-Res Presentation) */
  .ieee-fig {
    margin: 8px 0;
    break-inside: avoid;
    text-align: center;
  }
  .ieee-fig img {
    width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 3px auto;
    border: 0.3pt solid #ddd;
  }
  .ieee-fig-cap {
    font-size: 8pt;
    line-height: 1.15;
    text-align: justify;
  }
  .ieee-fig.span-all {
    column-span: all;
    margin: 14px 0 10px 0;
    text-align: center;
    break-inside: avoid;
  }
  .ieee-fig.span-all img {
    width: 100%;
    max-width: 7.25in;
    height: auto;
    display: block;
    margin: 0 auto 5px auto;
    border: 0.6pt solid #cbd5e1;
    border-radius: 4px;
    box-shadow: 0 1.5px 5px rgba(0,0,0,0.06);
    background: #ffffff;
  }
  .ieee-fig.span-all .ieee-fig-cap {
    font-size: 8.5pt;
    line-height: 1.25;
    text-align: justify;
    max-width: 7.25in;
    margin: 0 auto;
    padding: 0 4px;
  }

  /* Step-by-Step Calculation Walkthrough Boxes */
  .calc-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #1e3a8a;
    border-radius: 4px;
    padding: 7px 10px;
    margin: 10px 0;
    font-size: 8.3pt;
    line-height: 1.35;
    break-inside: avoid;
  }
  .calc-box.span-all {
    column-span: all;
    margin: 12px 0;
  }
  .calc-box-title {
    font-weight: bold;
    color: #1e3a8a;
    font-size: 8.6pt;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 5px;
    border-bottom: 0.6pt solid #cbd5e1;
    padding-bottom: 3px;
    display: flex;
    justify-content: space-between;
  }
  .calc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.0pt;
    margin-top: 3px;
  }
  .calc-table th {
    border-bottom: 1px solid #94a3b8;
    color: #1e293b;
    font-weight: bold;
    text-align: left;
    padding: 3.5px 4px;
    background: #f1f5f9;
  }
  .calc-table td {
    padding: 3px 4px;
    border-bottom: 0.5pt solid #e2e8f0;
    vertical-align: top;
  }
  .calc-table td.num {
    text-align: right;
    font-weight: 600;
    color: #0f172a;
    white-space: nowrap;
  }
  .calc-table td.eq {
    font-family: "Times New Roman", Times, serif;
    color: #334155;
  }"""

# Normalize newlines
html_text = html_text.replace("\r\n", "\n")
old_fig_css = old_fig_css.replace("\r\n", "\n")
new_fig_css = new_fig_css.replace("\r\n", "\n")

if old_fig_css in html_text:
    html_text = html_text.replace(old_fig_css, new_fig_css)
    print("Replaced CSS successfully.")
else:
    print("Warning: old_fig_css not found directly, checking regex...")

# 2. Convert all 6 figures to span-all
count_figs = html_text.count('<div class="ieee-fig">')
html_text = html_text.replace('<div class="ieee-fig">', '<div class="ieee-fig span-all">')
print(f"Converted {count_figs} figures to span-all full width.")

# 3. Calculation Box 1: First-Principles Electrochemical Derivation
box1_html = """    <!-- Calculation Box 1: Electrochemical Decomposition -->
    <div class="calc-box span-all">
      <div class="calc-box-title">
        <span>Calculation Box 1: First-Principles Electrochemical Walkthrough</span>
        <span>$i = 1.80\\text{ A/cm}^2,\\ 75^\\circ\\text{C},\\ 30\\text{ bar}$ (Nafion 115)</span>
      </div>
      <table class="calc-table">
        <thead>
          <tr>
            <th>Phenomenological Term</th>
            <th>Governing Equation &amp; Numerical Substitution</th>
            <th style="text-align:right;">Calculated Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Reversible Potential, $V_{\\text{rev}}$</td>
            <td class="eq">$$1.1867 + \\frac{8.314 \\times 348.15}{2 \\times 96485.3}\\ln\\left[\\frac{(30 - 0.3846)(1.0 - 0.3846)^{0.5}}{1.0}\\right]$$</td>
            <td class="num">$\\mathbf{1.2343\\text{ V}}$</td>
          </tr>
          <tr>
            <td>Anodic OER Activation, $\\eta_{\\text{act,an}}$</td>
            <td class="eq">$$\\frac{RT}{2 \\times 0.50 \\times F}\\text{arcsinh}\\left(\\frac{1.80}{2 \\times 4.14 \\times 10^{-5}}\\right) = 0.02999 \\times \\text{arcsinh}(21739)$$</td>
            <td class="num">$0.3253\\text{ V}$</td>
          </tr>
          <tr>
            <td>Cathodic HER Activation, $\\eta_{\\text{act,cat}}$</td>
            <td class="eq">$$\\frac{RT}{2 \\times 0.50 \\times F}\\text{arcsinh}\\left(\\frac{1.80}{2 \\times 0.0312}\\right) = 0.02999 \\times \\text{arcsinh}(28.85)$$</td>
            <td class="num">$0.0300\\text{ V}$</td>
          </tr>
          <tr>
            <td>Membrane Ohmic, $\\eta_{\\text{ohm,mem}}$</td>
            <td class="eq">$$1.80 \\times \\frac{0.0127\\text{ cm}}{0.1573\\text{ S/cm}} = 1.80 \\times 0.08074\\ \\Omega\\cdot\\text{cm}^2$$</td>
            <td class="num">$0.1453\\text{ V}$</td>
          </tr>
          <tr>
            <td>Contact Resistance, $\\eta_{\\text{ohm,elec}}$</td>
            <td class="eq">$$1.80 \\times R_{\\text{elec}} = 1.80 \\times 0.0320\\ \\Omega\\cdot\\text{cm}^2$$</td>
            <td class="num">$0.0576\\text{ V}$</td>
          </tr>
          <tr>
            <td>Mass Transport &amp; Bubbles, $\\eta_{\\text{conc}}$</td>
            <td class="eq">$$\\eta_{\\text{diff}}(0.0191\\text{ V}) + 0.014 \\times (1.80)^{1.65}(0.0271\\text{ V})$$</td>
            <td class="num">$0.0462\\text{ V}$</td>
          </tr>
          <tr style="background:#f1f5f9; font-weight:bold;">
            <td>Total Cell Potential, $V_{\\text{cell}}$</td>
            <td class="eq">$$V_{\\text{rev}} + \\eta_{\\text{act}} + \\eta_{\\text{ohm}} + \\eta_{\\text{conc}} = 1.2343 + 0.3553 + 0.2029 + 0.0462$$</td>
            <td class="num" style="color:#1e3a8a;">$\\mathbf{1.8387\\text{ V} \\approx 1.838\\text{ V}}$</td>
          </tr>
          <tr>
            <td>Faraday Efficiency, $\\eta_F(1.80)$</td>
            <td class="eq">$$\\text{Empirical fit to published gas crossover kinetics at nominal flux}$$</td>
            <td class="num">$\\mathbf{99.20\\%}$</td>
          </tr>
          <tr>
            <td>Stack Specific Energy, $\\text{SEC}_{\\text{stk}}$</td>
            <td class="eq">$$\\frac{26.590 \\times V_{\\text{cell}}}{\\eta_F} = \\frac{26.590 \\times 1.8387}{0.9920}$$</td>
            <td class="num">$\\mathbf{49.28\\text{ kWh/kg H}_2}$</td>
          </tr>
          <tr style="background:#f8fafc; font-weight:bold;">
            <td>System Specific Energy, $\\text{SEC}_{\\text{sys}}$</td>
            <td class="eq">$$\\frac{\\text{SEC}_{\\text{stk}}}{\\eta_{\\text{rect}}} + \\text{SEC}_{\\text{BOP}} = \\frac{49.28}{0.97} + 1.85$$</td>
            <td class="num" style="color:#047857;">$\\mathbf{52.62\\text{ kWh/kg H}_2}$</td>
          </tr>
        </tbody>
      </table>
    </div>
"""

target_t1_end = '<div class="tab-fn"><sup>a</sup>Nominal rating ($1.80\\text{ A/cm}^2$, $\\eta_{\\text{LHV}} = 63.34\\%$, $P_{\\text{dens}} = 3.308\\text{ W/cm}^2$).</div>\n    </div>'
if target_t1_end in html_text:
    html_text = html_text.replace(target_t1_end, target_t1_end + "\n\n" + box1_html)
    print("Inserted Calculation Box 1 after Table I.")
else:
    print("Warning: target_t1_end not found.")

# 4. Calculation Box 2: Worked Techno-Economic & LCOH Derivation
box2_html = """    <!-- Calculation Box 2: Techno-Economic & LCOH Life-Cycle Accounting -->
    <div class="calc-box span-all">
      <div class="calc-box-title">
        <span>Calculation Box 2: Worked Techno-Economic &amp; LCOH Derivation</span>
        <span>Optimal Knee-Point Hybrid Plant ($500\\text{ kg H}_2/\\text{h}$, $6$-Year Multi-Year Weather)</span>
      </div>
      <table class="calc-table">
        <thead>
          <tr>
            <th>Cost / Sizing Component</th>
            <th>Governing Formula &amp; Mathematical Accounting</th>
            <th style="text-align:right;">Subtotal / Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Electrolyzer Peak Power, $P_{\\text{PEM,peak}}$</td>
            <td class="eq">$$P_{\\text{PEM,nom}} \\times \\frac{i_{\\max} V_{\\text{cell}}(i_{\\max})}{i_{\\text{nom}} V_{\\text{cell}}(i_{\\text{nom}})} = 43.61\\text{ MW} \\times \\frac{1.764 \\times 1.832\\text{ V}}{1.800 \\times 1.838\\text{ V}}$$</td>
            <td class="num">$\\mathbf{42.58\\text{ MW}}$</td>
          </tr>
          <tr>
            <td>Electrolyzer Split-CAPEX, $C_{\\text{PEM}}$</td>
            <td class="eq">$$\\text{Stack }(43.61 \\times \\text{INR } 3.364\\text{ Cr}) + \\text{BOP/Rect }(42.58 \\times \\text{INR } 2.436\\text{ Cr})$$</td>
            <td class="num">$\\text{INR } 146.69\\text{ Cr} + \\text{INR } 103.73\\text{ Cr} = \\mathbf{250.42\\text{ Cr}}$</td>
          </tr>
          <tr>
            <td>Generation &amp; Storage CAPEX</td>
            <td class="eq">$$\\text{Wind }(48.02 \\times 6.20) + \\text{Solar }(72.77 \\times 3.60) + \\text{BESS }(86.86 \\times 1.65) + \\text{Tanks }(5.88) + \\text{Civil }(52.92)$$</td>
            <td class="num">$\\text{INR } 761.81\\text{ Cr}$</td>
          </tr>
          <tr style="background:#f1f5f9; font-weight:bold;">
            <td>Total Installed CAPEX, $\\text{CAPEX}_{\\text{tot}}$</td>
            <td class="eq">$$C_{\\text{wind}} + C_{\\text{solar}} + C_{\\text{PEM,stk}} + C_{\\text{PEM,bop}} + C_{\\text{batt}} + C_{\\text{tank}} + C_{\\text{civil}}$$</td>
            <td class="num" style="color:#1e3a8a;">$\\mathbf{INR\\ 1,012.23\\text{ Cr}}$</td>
          </tr>
          <tr>
            <td>Capital Recovery Factor, $\\text{CRF}$</td>
            <td class="eq">$$\\frac{r(1+r)^N}{(1+r)^N - 1} = \\frac{0.085(1.085)^{20}}{(1.085)^{20} - 1} \\quad (r = 8.5\\%,\\ N = 20\\text{ yr})$$</td>
            <td class="num">$\\mathbf{0.10567\\text{ yr}^{-1}}$</td>
          </tr>
          <tr>
            <td>Annualized Capital Cost, $\\text{AnnCAPEX}$</td>
            <td class="eq">$$\\text{CRF} \\times \\text{CAPEX}_{\\text{tot}} = 0.10567 \\times \\text{INR } 1,012.23\\text{ Cr}$$</td>
            <td class="num">$\\mathbf{INR\\ 106.96\\text{ Cr/year}}$</td>
          </tr>
          <tr>
            <td>Annual Fixed O&amp;M, $C_{\\text{O\\&M,fix}}$</td>
            <td class="eq">$$2.5\\% \\times (C_{\\text{wind}} + C_{\\text{solar}} + C_{\\text{BOP}} + C_{\\text{batt}}) + 3.0\\% \\times C_{\\text{stack}}$$</td>
            <td class="num">$\\text{INR } 27.42\\text{ Cr/year}$</td>
          </tr>
          <tr>
            <td>KERC Open-Access Wheeling, $C_{\\text{wheel}}$</td>
            <td class="eq">$$182,040\\text{ MWh/yr (delivered wind)} \\times \\text{INR } 0.58/\\text{kWh} \\quad (\\text{Tariff Order S/01/2024})$$</td>
            <td class="num">$\\text{INR } 13.11\\text{ Cr/year}$</td>
          </tr>
          <tr>
            <td>Stack Replacement &amp; Grid Backup</td>
            <td class="eq">$$\\text{Annualized Stack Replacement (INR } 2.41\\text{ Cr}) + \\text{Grid Backup (INR } 1.88\\text{ Cr}) - \\text{By-Product Credit (INR } 3.42\\text{ Cr})$$</td>
            <td class="num">$\\text{INR } 0.87\\text{ Cr/year}$</td>
          </tr>
          <tr style="background:#f1f5f9; font-weight:bold;">
            <td>Total Net Annualized Cost, $\\text{AnnCost}_{\\text{tot}}$</td>
            <td class="eq">$$\\text{AnnCAPEX} + C_{\\text{O\\&M,fix}} + C_{\\text{wheel}} + C_{\\text{repl}} + C_{\\text{grid}} - R_{\\text{curt}} = 106.96 + 27.42 + 13.11 + 0.87$$</td>
            <td class="num" style="color:#1e3a8a;">$\\mathbf{INR\\ 148.36\\text{ Cr/year}}$</td>
          </tr>
          <tr>
            <td>Annual Clean Hydrogen Output, $M_{\\text{H}_2}$</td>
            <td class="eq">$$500\\text{ kg/h} \\times 8,760\\text{ h} \\times (1 - 0.00622\\text{ LPSP}) \\times 1.025\\text{ (over-delivery)}$$</td>
            <td class="num">$\\mathbf{4,461,800\\text{ kg H}_2\\text{/year}}\\ (4,461.8\\text{ t/yr})$</td>
          </tr>
          <tr style="background:#f0fdf4; font-weight:bold; border-top:1.5px solid #16a34a;">
            <td>Levelized Cost of Hydrogen, $\\text{LCOH}$</td>
            <td class="eq">$$\\frac{\\text{Total Net Annualized Cost}}{\\text{Annual Hydrogen Produced}} = \\frac{\\text{INR } 148.36 \\times 10^7}{4,461,800\\text{ kg H}_2}$$</td>
            <td class="num" style="color:#15803d; font-size:9.2pt;">$\\mathbf{INR\\ 332.51/kg\\ H_2}\\ (\\mathbf{\\$3.96/kg})$</td>
          </tr>
        </tbody>
      </table>
    </div>
"""

target_t3_end = """            <td><strong>12.0</strong></td>
            <td>12.0</td>
            <td>12.0</td>
            <td>12.0</td>
            <td>12.0</td>
            <td>12.0</td>
          </tr>
        </tbody>
      </table>
    </div>"""

if target_t3_end in html_text:
    html_text = html_text.replace(target_t3_end, target_t3_end + "\n\n" + box2_html)
    print("Inserted Calculation Box 2 after Table III.")
else:
    print("Warning: target_t3_end not found.")

# 5. Calculation Box 3: Latent MSPM Statistics & Control Thresholds
box3_html = """    <!-- Calculation Box 3: Latent MSPM Statistics & Control Thresholds -->
    <div class="calc-box span-all">
      <div class="calc-box-title">
        <span>Calculation Box 3: Latent-Variable MSPM Detection Statistics &amp; Control Limits</span>
        <span>$9$-Sensor Process Monitoring &amp; Jackson–Mudholkar Thresholds ($\\alpha = 0.01$)</span>
      </div>
      <table class="calc-table">
        <thead>
          <tr>
            <th>MSPM Metric / Statistic</th>
            <th>Governing Mathematical Formulation</th>
            <th style="text-align:right;">Exact Value / Significance</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Latent Subspace Variance ($k = 3$)</td>
            <td class="eq">$$\\frac{\\lambda_1 + \\lambda_2 + \\lambda_3}{\\sum_{j=1}^9 \\lambda_j} \\times 100\\% = \\frac{4.182 + 2.814 + 1.647}{9.000} \\times 100\\%$$</td>
            <td class="num">$\\mathbf{96.03\\%\\text{ of total variance}}$</td>
          </tr>
          <tr>
            <td>Hotelling's $T^2$ Statistic, $T^2(t)$</td>
            <td class="eq">$$\\mathbf{z}_{\\text{std}}(t)^T \\mathbf{P}_3 \\mathbf{\\Lambda}_3^{-1} \\mathbf{P}_3^T \\mathbf{z}_{\\text{std}}(t) = \\sum_{j=1}^3 \\frac{t_j(t)^2}{\\lambda_j}$$</td>
            <td class="num">Latent score distance</td>
          </tr>
          <tr>
            <td>$T^2$ Upper Control Limit, $T_{0.99}^2$</td>
            <td class="eq">$$\\frac{k(N-1)}{N-k}F_{k, N-k}(0.01) \\approx \\chi_3^2(0.01) \\quad (\\text{Jackson \\&amp; Mudholkar 1979})$$</td>
            <td class="num" style="color:#1e3a8a;">$\\mathbf{T_{\\text{crit}}^2 = 11.58}$</td>
          </tr>
          <tr>
            <td>Residual $Q$/SPE Moments</td>
            <td class="eq">$$\\theta_1 = \\sum_{j=4}^9 \\lambda_j = 0.3570, \\quad \\theta_2 = \\sum_{j=4}^9 \\lambda_j^2 = 0.0241, \\quad \\theta_3 = \\sum_{j=4}^9 \\lambda_j^3 = 0.00178$$</td>
            <td class="num">$h_0 = 1 - \\frac{2\\theta_1 \\theta_3}{3\\theta_2^2} = \\mathbf{0.2695}$</td>
          </tr>
          <tr>
            <td>$Q$/SPE Control Limit, $Q_{0.99}$</td>
            <td class="eq">$$\\theta_1\\left[1 - \\frac{\\theta_2 h_0(1-h_0)}{\\theta_1^2} + \\frac{c_{\\alpha}\\sqrt{2\\theta_2 h_0^2}}{\\theta_1}\\right]^{1/h_0} \\quad (c_{0.01} = 2.3263)$$</td>
            <td class="num" style="color:#1e3a8a;">$\\mathbf{Q_{\\text{crit}} = 1.854}$</td>
          </tr>
          <tr>
            <td>Sensor Diagnostic Contribution, $\\text{Contr}_j^Q$</td>
            <td class="eq">$$\\frac{[\\mathbf{z}_{\\text{std},j}(t) - \\hat{\\mathbf{z}}_{\\text{std},j}(t)]^2}{Q(t)} \\times 100\\%$$</td>
            <td class="num">Normalized residual attribution</td>
          </tr>
          <tr style="background:#f1f5f9; font-weight:bold;">
            <td>Fault 1 Root-Cause Isolation</td>
            <td class="eq">$$\\text{Stack Temperature }(60.17\\%) + \\text{Cathode Pressure }(35.07\\%)$$</td>
            <td class="num" style="color:#047857;">$\\mathbf{95.24\\%\\text{ Thermal-Pressure Isolation}}$</td>
          </tr>
          <tr style="background:#f8fafc; font-weight:bold;">
            <td>Fault 2 Root-Cause Isolation</td>
            <td class="eq">$$\\text{Cell Voltage }(34.61\\%) + \\text{Cathode Pressure }(26.04\\%) + \\dot{m}_{\\text{H}_2}\\ (20.50\\%)$$</td>
            <td class="num" style="color:#047857;">$\\mathbf{81.15\\%\\text{ Overpotential-Flow Isolation}}$</td>
          </tr>
        </tbody>
      </table>
    </div>
"""

target_t4_end = """            <td class="left">--- Primary Sensor Isolation (Fault 2)</td>
            <td>Cell Voltage (34.6%)</td>
            <td>N/A</td>
          </tr>
        </tbody>
      </table>
    </div>"""

if target_t4_end in html_text:
    html_text = html_text.replace(target_t4_end, target_t4_end + "\n\n" + box3_html)
    print("Inserted Calculation Box 3 after Table IV.")
else:
    print("Warning: target_t4_end not found.")

html_path.write_text(html_text, encoding="utf-8")
print(f"Updated {html_path.name} successfully.")
