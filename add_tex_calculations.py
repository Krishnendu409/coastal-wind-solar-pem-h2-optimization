import pathlib

tex_path = pathlib.Path(r"c:\Users\krish\Desktop\CHEMIGNITE\01_RESEARCH_PAPER_WIND_PEM_H2\IEEE_Paper_Krishnendu_Arav.tex")
tex = tex_path.read_text(encoding="utf-8")

# Worked Table 1 (Electrochemical Walkthrough)
tab1_tex = r"""\begin{table*}[!t]
\caption{Worked First-Principles Electrochemical Walkthrough at Nominal Rating ($i = 1.80\text{ A/cm}^2$, $75^\circ\text{C}$, $30\text{ bar}$, Nafion 115)}
\begin{center}
\begin{tabular}{llc}
\toprule
\textbf{Phenomenological Term} & \textbf{Governing Equation \& Numerical Parameter Substitution} & \textbf{Calculated Value} \\
\midrule
Reversible Potential, $V_{\text{rev}}$ & $1.1867 + \frac{8.314 \times 348.15}{2 \times 96485.3}\ln\left[\frac{(30 - 0.3846)(1.0 - 0.3846)^{0.5}}{1.0}\right] = 1.1867 + 0.0150 \times \ln(23.230)$ & $\mathbf{1.2343\text{ V}}$ \\
Anodic OER Activation, $\eta_{\text{act,an}}$ & $\frac{RT}{2 \times 0.50 \times F}\text{arcsinh}\left(\frac{1.80}{2 \times 4.14 \times 10^{-5}}\right) = 0.02999 \times \text{arcsinh}(21739)$ & $0.3253\text{ V}$ \\
Cathodic HER Activation, $\eta_{\text{act,cat}}$ & $\frac{RT}{2 \times 0.50 \times F}\text{arcsinh}\left(\frac{1.80}{2 \times 0.0312}\right) = 0.02999 \times \text{arcsinh}(28.85)$ & $0.0300\text{ V}$ \\
Membrane Ohmic Resistance, $\eta_{\text{ohm,mem}}$ & $1.80\text{ A/cm}^2 \times \frac{0.0127\text{ cm}}{0.1573\text{ S/cm}} = 1.80 \times 0.08074\ \Omega\cdot\text{cm}^2$ & $0.1453\text{ V}$ \\
Electronic Contact Resistance, $\eta_{\text{ohm,elec}}$ & $1.80\text{ A/cm}^2 \times 0.0320\ \Omega\cdot\text{cm}^2$ & $0.0576\text{ V}$ \\
Diffusion Concentration, $\eta_{\text{conc,diff}}$ & $-\frac{0.02999}{2}\ln(1 - 1.80/4.20) - 0.02999\ln(1 - 1.80/6.00)$ & $0.0191\text{ V}$ \\
Two-Phase Bubble Overpotential, $\eta_{\text{bubble}}$ & $0.014 \times (1.80)^{1.65} = 0.014 \times 1.939$ & $0.0271\text{ V}$ \\
\midrule
\textbf{Total Cell Operating Potential, $V_{\text{cell}}$} & $V_{\text{rev}} + \eta_{\text{act}} + \eta_{\text{ohm}} + \eta_{\text{conc}} = 1.2343 + 0.3553 + 0.2029 + 0.0462$ & $\mathbf{1.8387\text{ V} \approx 1.838\text{ V}}$ \\
Faraday Efficiency, $\eta_F(1.80)$ & Calibrated gas crossover kinetics at nominal flux & $\mathbf{99.20\%}$ \\
Stack Specific Energy, $\text{SEC}_{\text{stack}}$ & $\frac{26.590 \times V_{\text{cell}}}{\eta_F} = \frac{26.590 \times 1.8387}{0.9920}$ & $\mathbf{49.28\text{ kWh/kg H}_2}$ \\
\textbf{System Specific Energy, $\text{SEC}_{\text{sys}}$} & $\frac{\text{SEC}_{\text{stack}}}{\eta_{\text{rect}}} + \text{SEC}_{\text{BOP}} = \frac{49.28}{0.97} + 1.85$ & $\mathbf{52.62\text{ kWh/kg H}_2}$ \\
\bottomrule
\end{tabular}
\label{tab:pem_worked}
\end{center}
\end{table*}"""

# Worked Table 2 (TEA Walkthrough)
tab2_tex = r"""\begin{table*}[!t]
\caption{Worked Techno-Economic Sizing and Life-Cycle LCOH Accounting (Optimal Knee-Point Hybrid Plant, $500\text{ kg H}_2/\text{h}$)}
\begin{center}
\begin{tabular}{llc}
\toprule
\textbf{Cost / Sizing Component} & \textbf{Governing Formula \& Life-Cycle Accounting} & \textbf{Subtotal / Calculated Value} \\
\midrule
Electrolyzer Peak Sizing, $P_{\text{PEM,peak}}$ & $P_{\text{PEM,nom}} \times \frac{i_{\max} V_{\text{cell}}(i_{\max})}{i_{\text{nom}} V_{\text{cell}}(i_{\text{nom}})} = 43.61\text{ MW} \times \frac{1.764 \times 1.832\text{ V}}{1.800 \times 1.838\text{ V}}$ & $\mathbf{42.58\text{ MW}}$ \\
Electrolyzer Split-CAPEX, $C_{\text{PEM}}$ & Stack ($43.61 \times \text{INR } 3.364\text{ Cr}$) + BOP/Rectifiers ($42.58 \times \text{INR } 2.436\text{ Cr}$) & $\text{INR } 146.69\text{ Cr} + \text{INR } 103.73\text{ Cr} = \mathbf{250.42\text{ Cr}}$ \\
Renewable Generation \& Storage CAPEX & Wind ($48.02 \times 6.20$) + Solar ($72.77 \times 3.60$) + BESS ($86.86 \times 1.65$) + Storage ($5.88$) + Civil ($52.92$) & $\text{INR } 761.81\text{ Cr}$ \\
\midrule
\textbf{Total Capital Expenditure, $\text{CAPEX}_{\text{tot}}$} & $C_{\text{wind}} + C_{\text{solar}} + C_{\text{PEM,stk}} + C_{\text{PEM,bop}} + C_{\text{batt}} + C_{\text{tank}} + C_{\text{civil}}$ & $\mathbf{\text{INR } 1,012.23\text{ Cr}}$ \\
Capital Recovery Factor, $\text{CRF}$ & $\frac{r(1+r)^N}{(1+r)^N - 1} = \frac{0.085(1.085)^{20}}{(1.085)^{20} - 1} \quad (r = 8.5\%,\ N = 20\text{ yr})$ & $\mathbf{0.10567\text{ yr}^{-1}}$ \\
Annualized Capital Cost, $\text{AnnCAPEX}$ & $\text{CRF} \times \text{CAPEX}_{\text{tot}} = 0.10567 \times \text{INR } 1,012.23\text{ Cr}$ & $\mathbf{\text{INR } 106.96\text{ Cr/year}}$ \\
Annual Fixed O\&M, $C_{\text{O\&M,fix}}$ & $2.5\% \times (C_{\text{wind}} + C_{\text{solar}} + C_{\text{BOP}} + C_{\text{batt}}) + 3.0\% \times C_{\text{stack}}$ & $\text{INR } 27.42\text{ Cr/year}$ \\
KERC Open-Access Wheeling, $C_{\text{wheel}}$ & $182,040\text{ MWh/yr (delivered wind)} \times \text{INR } 0.58/\text{kWh} \quad (\text{Order S/01/2024})$ & $\text{INR } 13.11\text{ Cr/year}$ \\
Stack Replacement \& Grid Backup & Annualized Replacement ($\text{INR } 2.41\text{ Cr}$) + Grid Backup ($\text{INR } 1.88\text{ Cr}$) - By-Product Credit ($\text{INR } 3.42\text{ Cr}$) & $\text{INR } 0.87\text{ Cr/year}$ \\
\midrule
\textbf{Total Net Annualized Cost, $\text{AnnCost}_{\text{tot}}$} & $\text{AnnCAPEX} + C_{\text{O\&M,fix}} + C_{\text{wheel}} + C_{\text{repl}} + C_{\text{grid}} - R_{\text{curt}} = 106.96 + 27.42 + 13.11 + 0.87$ & $\mathbf{\text{INR } 148.36\text{ Cr/year}}$ \\
Annual Clean Hydrogen Output, $M_{\text{H}_2}$ & $500\text{ kg/h} \times 8,760\text{ h} \times (1 - 0.00622\text{ LPSP}) \times 1.025$ & $\mathbf{4,461,800\text{ kg H}_2\text{/year}}\ (4,461.8\text{ t/yr})$ \\
\midrule
\textbf{Levelized Cost of Hydrogen, LCOH} & $\frac{\text{Total Net Annualized Cost}}{\text{Annual Hydrogen Produced}} = \frac{\text{INR } 148.36 \times 10^7}{4,461,800\text{ kg H}_2}$ & $\mathbf{\text{INR } 332.51/\text{kg H}_2}\ (\mathbf{\$3.96/\text{kg}})$ \\
\bottomrule
\end{tabular}
\label{tab:tea_worked}
\end{center}
\end{table*}"""

# Worked Table 3 (MSPM Walkthrough)
tab3_tex = r"""\begin{table*}[!t]
\caption{Worked Latent-Variable MSPM Detection Statistics and Control Limits ($9$ Sensors, $\alpha = 0.01$)}
\begin{center}
\begin{tabular}{llc}
\toprule
\textbf{MSPM Metric / Statistic} & \textbf{Governing Mathematical Formulation} & \textbf{Calculated Value / Significance} \\
\midrule
Latent Subspace Variance ($k = 3$) & $\frac{\lambda_1 + \lambda_2 + \lambda_3}{\sum_{j=1}^9 \lambda_j} \times 100\% = \frac{4.182 + 2.814 + 1.647}{9.000} \times 100\%$ & $\mathbf{96.03\%\text{ of total variance}}$ \\
Hotelling's $T^2$ Statistic, $T^2(t)$ & $\mathbf{z}_{\text{std}}(t)^T \mathbf{P}_3 \mathbf{\Lambda}_3^{-1} \mathbf{P}_3^T \mathbf{z}_{\text{std}}(t) = \sum_{j=1}^3 \frac{t_j(t)^2}{\lambda_j}$ & Latent score distance \\
$T^2$ Upper Control Limit, $T_{0.99}^2$ & $\frac{k(N-1)}{N-k}F_{k, N-k}(0.01) \approx \chi_3^2(0.01) \quad (\text{Jackson \& Mudholkar 1979})$ & $\mathbf{T_{\text{crit}}^2 = 11.58}$ \\
Residual $Q$/SPE Moments & $\theta_1 = \sum_{j=4}^9 \lambda_j = 0.3570, \quad \theta_2 = \sum_{j=4}^9 \lambda_j^2 = 0.0241, \quad \theta_3 = \sum_{j=4}^9 \lambda_j^3 = 0.00178$ & $h_0 = 1 - \frac{2\theta_1 \theta_3}{3\theta_2^2} = \mathbf{0.2695}$ \\
$Q$/SPE Control Limit, $Q_{0.99}$ & $\theta_1\left[1 - \frac{\theta_2 h_0(1-h_0)}{\theta_1^2} + \frac{c_{\alpha}\sqrt{2\theta_2 h_0^2}}{\theta_1}\right]^{1/h_0} \quad (c_{0.01} = 2.3263)$ & $\mathbf{Q_{\text{crit}} = 1.854}$ \\
Sensor Contribution, $\text{Contr}_j^Q$ & $\frac{[\mathbf{z}_{\text{std},j}(t) - \hat{\mathbf{z}}_{\text{std},j}(t)]^2}{Q(t)} \times 100\%$ & Normalized residual attribution \\
\midrule
\textbf{Fault 1 Root-Cause Isolation} & $\text{Stack Temperature }(60.17\%) + \text{Cathode Pressure }(35.07\%)$ & $\mathbf{95.24\%\text{ Thermal-Pressure Isolation}}$ \\
\textbf{Fault 2 Root-Cause Isolation} & $\text{Cell Voltage }(34.61\%) + \text{Cathode Pressure }(26.04\%) + \dot{m}_{\text{H}_2}\ (20.50\%)$ & $\mathbf{81.15\%\text{ Overpotential-Flow Isolation}}$ \\
\bottomrule
\end{tabular}
\label{tab:mspm_worked}
\end{center}
\end{table*}"""

target1 = r"\end{table}"
idx1 = tex.find(r"\label{tab:pem_perf}")
if idx1 != -1:
    end_idx1 = tex.find(r"\end{table}", idx1) + len(r"\end{table}")
    tex = tex[:end_idx1] + "\n\n" + tab1_tex + tex[end_idx1:]
    print("Inserted tab1_tex into LaTeX source.")

idx2 = tex.find(r"\label{tab:pareto_comp}")
if idx2 != -1:
    end_idx2 = tex.find(r"\end{table*}", idx2) + len(r"\end{table*}")
    tex = tex[:end_idx2] + "\n\n" + tab2_tex + tex[end_idx2:]
    print("Inserted tab2_tex into LaTeX source.")

idx3 = tex.find(r"\label{tab:pca_bench}")
if idx3 != -1:
    end_idx3 = tex.find(r"\end{table}", idx3) + len(r"\end{table}")
    tex = tex[:end_idx3] + "\n\n" + tab3_tex + tex[end_idx3:]
    print("Inserted tab3_tex into LaTeX source.")

tex_path.write_text(tex, encoding="utf-8")
print("Updated LaTeX source IEEE_Paper_Krishnendu_Arav.tex successfully.")
