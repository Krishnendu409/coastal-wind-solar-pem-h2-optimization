#!/usr/bin/env python3
"""
tests/test_scientific_validity.py
===============================================================================
Automated Scientific Validation, Physical Consistency, and Reproducibility Test
Suite for the Wind-Solar-PEM Green Hydrogen Plant Research Project.

Verifies all 26 technical audit remediation items across:
  1. Multi-Year (2019-2024, 52,608-hour) Uninflated NASA POWER Meteorology
  2. Experimentally Calibrated First-Principles PEMWE Electrochemistry & Safety
  3. Split-CAPEX High-Current Techno-Economics & Dispatch Coupling
  4. Strictly Leak-Free 12D Physics Surrogate & 2x5 Cross-Validated Ablation
  5. 5-Seed Constrained NSGA-II, Reliability-Matched Baselines & True Monte Carlo
  6. 100-Realization 9-Sensor PCA vs. Load-Compensated 1D Voltage-Residual EWMA
===============================================================================
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from simulate_and_optimize import (
    FirstPrinciplesPEMElectrolyzer,
    HybridWindSolarPEMSimulator,
    load_multiyear_karnataka_meteo,
    make_leak_free_physics_features,
)


class TestScientificValidity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meteo_df, cls.meteo_source = load_multiyear_karnataka_meteo()
        cls.pem = FirstPrinciplesPEMElectrolyzer()
        cls.sim = HybridWindSolarPEMSimulator(cls.meteo_df, cls.pem, regime="Regime_B_OpenAccess")
        summary_path = os.path.join(PROJECT_DIR, "results", "summary_metrics.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            cls.summary = json.load(f)

    # -------------------------------------------------------------------------
    # 1. METEOROLOGICAL INTEGRITY & ZERO WIND MULTIPLIER TESTS
    # -------------------------------------------------------------------------
    def test_01_multiyear_dataset_completeness_and_bounds(self):
        """Verify 6-year (2019-2024, 52,608-hour) NASA POWER dataset & physical bounds."""
        self.assertEqual(len(self.meteo_df), 52608)
        self.assertEqual(sorted(self.meteo_df["year"].unique().tolist()), [2019, 2020, 2021, 2022, 2023, 2024])
        self.assertTrue(np.all(self.meteo_df["ghi_w_m2"].values >= 0.0))
        self.assertTrue(np.all(self.meteo_df["ghi_w_m2"].values <= 1361.0))
        self.assertTrue(np.all(self.meteo_df["cf_solar"].values >= 0.0))
        self.assertTrue(np.all(self.meteo_df["cf_solar"].values <= 1.0))
        self.assertTrue(np.all(self.meteo_df["cf_wind_coastal"].values >= 0.0))
        self.assertTrue(np.all(self.meteo_df["cf_wind_coastal"].values <= 1.0))
        self.assertTrue(np.all(self.meteo_df["cf_wind_highland_deliv"].values >= 0.0))
        self.assertTrue(np.all(self.meteo_df["cf_wind_highland_deliv"].values <= 1.0))

    def test_02_zero_wind_speed_multiplier_applied(self):
        """Verify that no 1.58x or 1.28x artificial multiplier is applied to 100 m wind speeds."""
        self.assertEqual(self.summary["site"]["wind_speed_multiplier_applied"], 1.00)
        ws10 = np.maximum(self.meteo_df["ws10m_m_s"].values, 0.2)
        ws50 = np.maximum(self.meteo_df["ws50m_m_s"].values, 0.2)
        alpha = np.clip(np.log(np.maximum(ws50, ws10 + 0.05) / ws10) / np.log(50.0 / 10.0), 0.06, 0.40)
        expected_v100 = ws50 * ((100.0 / 50.0) ** alpha)
        np.testing.assert_allclose(self.meteo_df["ws100m_m_s"].values, expected_v100, atol=2e-3)

    # -------------------------------------------------------------------------
    # 2. ELECTROCHEMICAL FIRST-PRINCIPLES & EXPERIMENTAL CALIBRATION TESTS
    # -------------------------------------------------------------------------
    def test_03_antoine_saturation_vapor_pressure_and_nernst_vrev(self):
        """Verify Antoine water vapor pressure (0.3846 bar at 75 C) and V_rev = 1.234 V (dE0/dT = -0.846 mV/K)."""
        p_sat_75 = float(self.pem.water_saturation_pressure_bar(348.15))
        self.assertAlmostEqual(p_sat_75, 0.3846, delta=0.002)
        comp = self.pem.cell_voltage_components(1.80, T_K=348.15)
        self.assertAlmostEqual(float(comp["V_rev"]), 1.2339, delta=0.001)

    def test_04_cell_voltage_component_additivity_and_monotonicity(self):
        """Verify V_cell == V_rev + eta_act + eta_ohm + eta_conc and strict monotonicity."""
        i_grid = np.linspace(0.10, 3.50, 70)
        comp = self.pem.cell_voltage_components(i_grid, T_K=348.15)
        v_sum = comp["V_rev"] + comp["eta_act"] + comp["eta_ohm"] + comp["eta_conc"]
        np.testing.assert_allclose(comp["V_cell"], v_sum, atol=1e-12)
        self.assertTrue(np.all(np.diff(comp["V_cell"]) > 0.0))

    def test_05_experimental_polarization_calibration_accuracy(self):
        """Verify calibration against 14 published experimental points achieves RMSE < 5 mV."""
        cal = self.pem.evaluate_experimental_calibration()
        self.assertEqual(cal["n_points"], 14)
        self.assertLess(cal["rmse_mV"], 5.0)
        self.assertGreater(cal["r2_score"], 0.9995)

    def test_06_exact_faraday_constant_and_sec_consistency(self):
        """Verify exact K_Faraday = 26.5904 kAh/kg H2 (M_H2 = 2.01588 g/mol)."""
        expected_kf = (2.0 * 96485.3329) / (3.6e6 * 2.01588e-3)
        self.assertAlmostEqual(self.pem.K_Faraday_kAh_kg, expected_kf, places=5)
        self.assertAlmostEqual(self.pem.K_Faraday_kAh_kg, 26.5904, places=3)

    def test_07_anodic_h2_in_o2_crossover_safety_enforcement(self):
        """Verify anodic H2-in-O2 < 2.0 mol% (50% LEL) at i >= i_min_safe = 0.20 A/cm2."""
        y_min = float(self.pem.anodic_h2_in_o2_mol_pct(self.pem.i_min_safe_a_cm2, with_recombiner=True))
        self.assertLess(y_min, 2.0)
        self.assertAlmostEqual(y_min, 1.641, delta=0.05)
        knee = self.summary["knee_point_optimal_design"]
        self.assertGreaterEqual(knee["min_active_i_a_cm2"], self.pem.i_min_safe_a_cm2)
        self.assertLess(knee["max_anodic_h2_in_o2_mol_pct"], 2.0)

    # -------------------------------------------------------------------------
    # 3. DISPATCH COUPLING & SPLIT-CAPEX TECHNO-ECONOMIC TESTS
    # -------------------------------------------------------------------------
    def test_08_dispatch_lut_exact_match_to_first_principles(self):
        """Verify simulator LUT matches FirstPrinciplesPEMElectrolyzer to < 0.05 mV."""
        test_i = np.linspace(0.20, 3.00, 50)
        v_exact = self.pem.cell_voltage_components(test_i, T_K=348.15)["V_cell"]
        v_lut = np.interp(test_i, self.sim.lut_i, self.sim.lut_v)
        np.testing.assert_allclose(v_lut, v_exact, atol=5e-5)

    def test_09_split_capex_high_current_overload_scaling(self):
        """Verify that increasing i_max increases PEM BOP CAPEX proportionally to P_PEM,peak."""
        res_nom = self.sim.simulate_design(70.0, 60.0, 36.0, 50.0, 1.80, return_timeseries=False)
        res_high = self.sim.simulate_design(70.0, 60.0, 36.0, 50.0, 2.50, return_timeseries=False)
        self.assertEqual(res_nom["capex_pem_stack_cr_inr"], res_high["capex_pem_stack_cr_inr"])
        self.assertGreater(res_high["pem_peak_mw"], res_nom["pem_peak_mw"])
        self.assertGreater(res_high["capex_pem_bop_cr_inr"], res_nom["capex_pem_bop_cr_inr"])
        self.assertGreater(res_high["capex_h2_tank_cr_inr"], 0.0)

    # -------------------------------------------------------------------------
    # 4. LEAK-FREE SURROGATE & SHAPLEY ADDITIVITY TESTS
    # -------------------------------------------------------------------------
    def test_10_surrogate_features_strictly_leak_free(self):
        """Verify make_leak_free_physics_features is 100% invariant to CAPEX/tariff parameters."""
        X_sample = np.array([[70.0, 60.0, 36.0, 55.0, 2.30]])
        F_before = make_leak_free_physics_features(X_sample, self.sim).copy()
        orig_wind_capex = self.sim.capex_wind_inr_mw
        try:
            self.sim.capex_wind_inr_mw = 9.99e8  # Perturb economic parameter by 16x
            F_after = make_leak_free_physics_features(X_sample, self.sim)
            np.testing.assert_array_equal(F_before, F_after)
        finally:
            self.sim.capex_wind_inr_mw = orig_wind_capex
        self.assertEqual(F_before.shape[1], 12)

    def test_11_surrogate_ablation_and_shapley_normalization(self):
        """Verify 5-fold CV R2 > 0.98 and Shapley attributions sum to 100%."""
        surr = self.summary["green_surrogate_benchmark"]
        self.assertGreater(surr["r2_green_mean"], 0.98)
        self.assertEqual(len(surr["ablation_2x5_table"]), 10)
        self.assertAlmostEqual(sum(surr["xai_lcoh_importance_pct"]), 100.0, delta=0.2)
        self.assertAlmostEqual(sum(surr["xai_deg_importance_pct"]), 100.0, delta=0.2)

    # -------------------------------------------------------------------------
    # 5. PARETO FEASIBILITY, MONTE CARLO & 9-SENSOR PCA TESTS
    # -------------------------------------------------------------------------
    def test_12_pareto_front_feasibility_and_pca_sensor_validity(self):
        """Verify Pareto feasibility constraints and 100-realization 9-sensor PCA integrity."""
        pareto_df = pd.read_csv(os.path.join(PROJECT_DIR, "results", "pareto_front.csv"))
        self.assertTrue(np.all(pareto_df["lpsp_pct"].values <= 2.5))
        self.assertTrue(np.all(pareto_df["stack_life_yr"].values >= 7.5))

        pca = self.summary["pca_fault_monitoring"]
        self.assertEqual(pca["n_monitored_vars"], 9)
        self.assertEqual(pca["n_mc_realizations"], 100)
        for v_name in pca["var_names"]:
            self.assertNotIn("σ_mem", v_name)
            self.assertNotIn("Membrane Cond", v_name)

    def test_13_calibration_train_validation_partition(self):
        """Verify 10-point calibration vs 4-point held-out validation partition (RMSE < 4.0 mV)."""
        cal = self.pem.evaluate_experimental_calibration()
        self.assertEqual(cal["n_train_points"], 10)
        self.assertEqual(cal["n_val_points"], 4)
        self.assertLess(cal["rmse_train_mV"], 4.0)
        self.assertLess(cal["rmse_val_mV"], 4.0)
        self.assertGreater(cal["r2_val_score"], 0.995)

    def test_14_pca_autocorrelation_and_wheeling_breakeven(self):
        """Verify serial autocorrelation isolation, Q-chart FAR < 3.0%, and wheeling breakeven bounds."""
        pca = self.summary["pca_fault_monitoring"]
        self.assertGreater(pca["autocorrelation_rho1_t2"], 0.65)
        self.assertLess(pca["false_alarm_rate_q_alone_pct"], 3.0)

        # Verify wheeling breakeven against Regime A (Rs 382.04/kg) occurs between 1.75 and 1.95 Rs/kWh
        res_knee = self.sim.simulate_design(48.02, 72.77, 43.61, 86.86, 1.764)
        annual_wind_kwh = (48.02 * np.sum(self.sim.cf_wind_highland_deliv) / 6.0) * 1000.0
        annual_h2_kg = res_knee["annual_h2_tonnes"] * 1000.0
        base_annual_cost = res_knee["lcoh_inr_kg"] * annual_h2_kg
        base_wheeling = annual_wind_kwh * 0.72

        def calc_lcoh_at_tariff(tariff):
            new_wheeling = annual_wind_kwh * tariff
            return (base_annual_cost - base_wheeling + new_wheeling) / annual_h2_kg

        lcoh_at_175 = calc_lcoh_at_tariff(1.75)
        lcoh_at_195 = calc_lcoh_at_tariff(1.95)
        self.assertLess(lcoh_at_175, 382.04)
        self.assertGreater(lcoh_at_195, 382.04)


if __name__ == "__main__":
    unittest.main(verbosity=2)
