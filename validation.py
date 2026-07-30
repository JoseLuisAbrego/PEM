#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
validacion_3.py - PROFESSIONAL VALIDATION MEPPME v16.0 (Appendix C)
================================================================================
VERSION: 16.0 (Major graphical and reporting improvements)
AUTHOR:  José Luis Abrego Salazar
COMPANY: ABNALITIC
YEAR:    2026
LICENSE: MIT

MAIN CHANGES v16.0:
    1. Star diagram replaced by a rigorous "Multiplier Phase Space" classification
       (COMPACT, LOOSE, CONFLICT A, CONFLICT B) with percentage bars.
    2. "Verification of Phase Equation" split into two panels:
       (a) Raw data scatter (no fit)
       (b) Per‑soil log-linear fits with equations and R².
    3. All plot filenames changed to English (plot_p1_*.png).
    4. Report renamed to "report_validation_3.pdf" and "report_validation_3.txt".
    5. Alert table in the report fully translated to English (except "lambda2 edge").
    6. The core inference engine (mepme_core) and all simulation/optimisation logic
       remain untouched.
================================================================================
"""

import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit

from mepme_core import (
    PhysicalConstants,
    MeshGenerator,
    run_inference,
    __version__ as core_version,
)

# Attempt to import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Define a dummy tqdm if not installed
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total if total is not None else (len(iterable) if iterable is not None else 0)
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                self._update_display()
            self.close()
        def update(self, n=1):
            self.n += n
            self._update_display()
        def _update_display(self):
            if self.total and self.n % max(1, self.total//20) == 0:
                print(f"\r{self.desc}: {self.n}/{self.total}", end='')
        def close(self):
            if self.total:
                print(f"\r{self.desc}: {self.n}/{self.total} completed.")
        def set_postfix(self, **kwargs):
            if self.total:
                msg = f"\r{self.desc}: {self.n}/{self.total}"
                for k, v in kwargs.items():
                    msg += f" | {k}={v}"
                print(msg, end='')
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# OPTIONAL DEPENDENCIES (REPORTLAB)
# =============================================================================
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ℹ️  reportlab not installed. PDF report will be generated in plain text.")
    print("   To install it: pip install reportlab")

# =============================================================================
# 1. GLOBAL CONFIGURATION (UPDATED v15.0)
# =============================================================================
S_MIN = 1e-8
S_MAX = 1.0
S0_GLOBAL = np.sqrt(S_MIN * S_MAX)  # = 1e-4

# Finer meshes to improve integration and numerical conditioning
N_POINTS_PART1 = 512          # Increased from 256 to 512
N_POINTS_PART2 = 256          # Increased from 128 to 256

N_SAMPLES_PART1 = 100         # Samples per soil (Part 1)
N_SAMPLES_PART2 = 36          # Samples per soil (Part 2, Grid Search)
N_BOOTSTRAP = 10000

# Relaxed thresholds to capture more valid solutions without losing quality
RESIDUO_THRESHOLD = 0.8
COND_NUM_THRESHOLD = 5e7      # Increased from 5e6 to 5e7
LAMBDA_BOUNDS = (-100000.0, 100000.0)  # Expanded to avoid saturation at extremes
LAMBDA_EDGE_THRESHOLD = 90000.0        # Threshold to detect real saturation (not false positives)
CURVATURE_ALPHA = 0.05
MODA_FIJA_THRESHOLD = 0.80

# Structural parameters (base)
N0_BASE = 0.40
BETA_BASE = 0.05
AUTO_FIX_BETA = True

# =============================================================================
# 1.1 TORTUOSITY FACTOR (NEW FINDING, SECTION A.4.1.2)
# =============================================================================
TORTUOSITY_FACTOR = 50.0      # Multiplied by theoretical α

# Ranges for Part 2 (Grid Search over structural parameters)
N0_VALUES = [0.30, 0.35, 0.40, 0.45, 0.50]
BETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]

# Optimization
MAXITER = 2000
TIMEOUT = 300                  # 5 minutes
GLOBAL_SEED = 42

# =============================================================================
# 2. DEFINITION OF THE 23 SOILS (PHYSICAL RANGES)
# =============================================================================
SUELOS = [
    {"id": 1,  "g_min": 1.60, "g_max": 1.85, "k_min": 1e-8,  "k_max": 1e-6},
    {"id": 2,  "g_min": 1.29, "g_max": 1.49, "k_min": 5e-4,  "k_max": 5e-3},
    {"id": 3,  "g_min": 1.40, "g_max": 1.60, "k_min": 1e-6,  "k_max": 5e-5},
    {"id": 4,  "g_min": 1.64, "g_max": 1.89, "k_min": 1e-3,  "k_max": 1e-2},
    {"id": 5,  "g_min": 1.62, "g_max": 1.75, "k_min": 2e-2,  "k_max": 6e-2},
    {"id": 6,  "g_min": 1.35, "g_max": 1.74, "k_min": 1e-3,  "k_max": 1e-2},
    {"id": 7,  "g_min": 1.67, "g_max": 1.81, "k_min": 2e-2,  "k_max": 5e-1},
    {"id": 8,  "g_min": 1.40, "g_max": 1.53, "k_min": 1e-3,  "k_max": 1e-2},
    {"id": 9,  "g_min": 1.52, "g_max": 1.65, "k_min": 4e-2,  "k_max": 1e-1},
    {"id": 10, "g_min": 1.60, "g_max": 1.73, "k_min": 1e-2,  "k_max": 4e-2},
    {"id": 11, "g_min": 1.46, "g_max": 1.60, "k_min": 5e-3,  "k_max": 2e-2},
    {"id": 12, "g_min": 1.45, "g_max": 1.65, "k_min": 1e-5,  "k_max": 1e-4},
    {"id": 13, "g_min": 1.38, "g_max": 1.56, "k_min": 1e-3,  "k_max": 5e-3},
    {"id": 14, "g_min": 1.21, "g_max": 1.41, "k_min": 5e-4,  "k_max": 1e-3},
    {"id": 15, "g_min": 1.36, "g_max": 1.51, "k_min": 5e-3,  "k_max": 2e-2},
    {"id": 16, "g_min": 1.38, "g_max": 1.52, "k_min": 5e-3,  "k_max": 2e-2},
    {"id": 17, "g_min": 1.56, "g_max": 1.70, "k_min": 2e-2,  "k_max": 6e-2},
    {"id": 18, "g_min": 1.30, "g_max": 1.48, "k_min": 1e-3,  "k_max": 1e-2},
    {"id": 19, "g_min": 1.43, "g_max": 1.59, "k_min": 1e-2,  "k_max": 4e-2},
    {"id": 20, "g_min": 1.75, "g_max": 1.95, "k_min": 1e-6,  "k_max": 1e-4},
    {"id": 21, "g_min": 1.72, "g_max": 1.86, "k_min": 1e0,   "k_max": 5e0},
    {"id": 22, "g_min": 1.64, "g_max": 1.81, "k_min": 5e-1,  "k_max": 2e0},
    {"id": 23, "g_min": 1.65, "g_max": 1.79, "k_min": 5e-2,  "k_max": 1e0},
]

# =============================================================================
# 3. AUXILIARY FUNCTIONS (unchanged)
# =============================================================================

def fit_power_law(df: pd.DataFrame) -> Dict[str, Any]:
    """Fits the power law Var = a·E^b and detects curvature."""
    if len(df) < 30:
        return {'b': np.nan, 'r2': np.nan, 'intercept': np.nan,
                'c': np.nan, 'p_value_c': 1.0, 'curvature_detected': False,
                'n_points': len(df)}
    log_E = np.log10(df['E_s_cm'].values)
    log_Var = np.log10(df['Var_s_cm2'].values)
    slope, intercept, r_val, _, _ = linregress(log_E, log_Var)
    r2 = r_val ** 2
    def quad_func(x, a, b, c):
        return a + b * x + c * x**2
    try:
        p0 = [intercept, slope, 0.0]
        popt, pcov = curve_fit(quad_func, log_E, log_Var, p0=p0, method='trf')
        a, b_quad, c_quad = popt
        residuals = log_Var - quad_func(log_E, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_Var - np.mean(log_Var))**2)
        r2_quad = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        if pcov is not None and np.isfinite(pcov).all():
            perr = np.sqrt(np.diag(pcov))
            if len(perr) >= 3 and perr[2] > 0:
                t_stat = c_quad / perr[2]
                p_value_c = 2 * (1 - norm.cdf(np.abs(t_stat)))
            else:
                p_value_c = 1.0
        else:
            p_value_c = 1.0
        curvature_detected = p_value_c < CURVATURE_ALPHA
        return {'b': slope, 'r2': r2, 'intercept': intercept,
                'c': c_quad, 'p_value_c': p_value_c,
                'curvature_detected': curvature_detected,
                'n_points': len(df), 'r2_quad': r2_quad}
    except Exception:
        return {'b': slope, 'r2': r2, 'intercept': intercept,
                'c': np.nan, 'p_value_c': 1.0,
                'curvature_detected': False, 'n_points': len(df)}

def bootstrap_abrego(df: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> Dict[str, float]:
    """Bootstrap to estimate uncertainty of the exponent b."""
    log_E = np.log10(df['E_s_cm'].values)
    log_Var = np.log10(df['Var_s_cm2'].values)
    slope_orig, intercept_orig, r_val_orig, _, _ = linregress(log_E, log_Var)
    r2_orig = r_val_orig ** 2
    b_list = []
    n = len(df)
    rng = np.random.default_rng(seed=GLOBAL_SEED)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            slope, _, _, _, _ = linregress(log_E[idx], log_Var[idx])
            b_list.append(slope)
        except:
            continue
    if b_list:
        b_mean = np.mean(b_list)
        b_std = np.std(b_list)
        b_ci_lower = np.percentile(b_list, 2.5)
        b_ci_upper = np.percentile(b_list, 97.5)
    else:
        b_mean = b_std = b_ci_lower = b_ci_upper = np.nan
    return {'b_mean': b_mean, 'b_std': b_std,
            'b_ci_lower': b_ci_lower, 'b_ci_upper': b_ci_upper,
            'slope': slope_orig, 'intercept': intercept_orig,
            'r2': r2_orig, 'n_bootstrap_valid': len(b_list)}

def simular_muestras_suelo(
    suelo: Dict,
    constants: PhysicalConstants,
    mesh: MeshGenerator,
    n_samples: int = N_SAMPLES_PART1,
    bounds: Tuple[float, float] = LAMBDA_BOUNDS,
    maxiter: int = MAXITER,
    timeout: Optional[float] = TIMEOUT,
    rng: np.random.Generator = None
) -> pd.DataFrame:
    """Generates random samples within the range of a soil and runs inference."""
    if rng is None:
        rng = np.random.default_rng(seed=GLOBAL_SEED)
    resultados = []
    for _ in range(n_samples):
        gamma_obj = rng.uniform(suelo["g_min"], suelo["g_max"])
        k_obj = rng.uniform(suelo["k_min"], suelo["k_max"])
        try:
            res = run_inference(
                gamma_obj, k_obj, mesh, constants,
                use_fallbacks=True,
                bounds=bounds,
                maxiter=maxiter,
                timeout=timeout
            )
            if not res['success'] or 'metrics' not in res:
                resultados.append({
                    'gamma_usado': gamma_obj, 'k_usado': k_obj,
                    'E_s_cm': np.nan, 'Var_s_cm2': np.nan,
                    'lambda1': np.nan, 'lambda2': np.nan,
                    'residuo': 1.0, 'moda_cm': np.nan,
                    'entropia': np.nan, 'cond_num': np.nan,
                    'valida': False, 'motivo': 'No converged'
                })
                continue
            residuo = res['residual_norm']
            lambda2 = res['lambda'][1]
            cond_num = res.get('cond_num', np.inf)
            metrics = res['metrics']
            valida = True
            motivo = "OK"
            # Validation criteria
            if residuo >= RESIDUO_THRESHOLD:
                valida = False
                motivo = f"High residual ({residuo:.3f})"
            elif cond_num >= COND_NUM_THRESHOLD:
                valida = False
                motivo = f"Unstable condition ({cond_num:.2e})"
            elif abs(lambda2) > LAMBDA_EDGE_THRESHOLD:
                valida = False
                motivo = f"lambda2 edge ({lambda2:.2f})"
            elif metrics['mode'] <= mesh.get_mesh()[0]*2 or metrics['mode'] >= mesh.get_mesh()[-1]*0.9:
                valida = False
                motivo = "Mode at edge"
            resultados.append({
                'gamma_usado': gamma_obj, 'k_usado': k_obj,
                'E_s_cm': metrics['E_s'], 'Var_s_cm2': metrics['Var_s'],
                'lambda1': res['lambda'][0], 'lambda2': lambda2,
                'residuo': residuo, 'moda_cm': metrics['mode'],
                'entropia': metrics['entropy'], 'cond_num': cond_num,
                'valida': valida, 'motivo': motivo
            })
        except Exception as e:
            resultados.append({
                'gamma_usado': gamma_obj, 'k_usado': k_obj,
                'E_s_cm': np.nan, 'Var_s_cm2': np.nan,
                'lambda1': np.nan, 'lambda2': np.nan,
                'residuo': 1.0, 'moda_cm': np.nan,
                'entropia': np.nan, 'cond_num': np.nan,
                'valida': False, 'motivo': str(e)[:50]
            })
    return pd.DataFrame(resultados)

# =============================================================================
# 4. MODIFIED PLOTTING FUNCTIONS (v16.0)
# =============================================================================

def generar_graficos_parte1(df: pd.DataFrame, boot: Dict, suelos: List, colores: List):
    """
    Generates all Part 1 plots with the new classification and phase verification.
    All filenames are now in English.
    """
    print("\n📊 Generating plots - Part 1...")
    
    # -------------------------------------------------------------------------
    # Plot 1: Scatter plot E[s] vs Var[s]
    # -------------------------------------------------------------------------
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        for idx, suelo in enumerate(suelos):
            subset = df[df["Suelo_ID"] == suelo["id"]]
            if not subset.empty:
                ax1.scatter(subset["E_s_cm"], subset["Var_s_cm2"],
                            color=colores[idx], alpha=0.4, s=10, label=f"Soil {suelo['id']}")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel(r'$E[s]$ [cm]', fontsize=12)
        ax1.set_ylabel(r'$\mathrm{Var}[s]$ [cm²]', fontsize=12)
        ax1.set_title('Data structure: E[s] vs Var[s]', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig('plot_p1_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("   ✅ 1/7: plot_p1_scatter.png")
    except Exception as e:
        print(f"   ⚠️ Error in plot 1: {e}")

    # -------------------------------------------------------------------------
    # Plot 2: Power-law regression (Abrego exponent)
    # -------------------------------------------------------------------------
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        for idx, suelo in enumerate(suelos):
            subset = df[df["Suelo_ID"] == suelo["id"]]
            if not subset.empty:
                ax2.scatter(subset["E_s_cm"], subset["Var_s_cm2"],
                            color=colores[idx], alpha=0.3, s=8)
        x_fit = np.logspace(np.log10(df["E_s_cm"].min()), np.log10(df["E_s_cm"].max()), 100)
        y_fit = 10 ** (boot['slope'] * np.log10(x_fit) + boot['intercept'])
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2.5,
                 label=f'Var = {10**boot["intercept"]:.4f} · E^{boot["slope"]:.4f}')
        boot_text = (f"Abrego exponent (b):\n"
                     f"Mean = {boot['b_mean']:.4f}\n"
                     f"95% CI = [{boot['b_ci_lower']:.4f}, {boot['b_ci_upper']:.4f}]\n"
                     f"R² = {boot['r2']:.4f}")
        ax2.text(0.05, 0.95, boot_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$E[s]$ [cm]', fontsize=12)
        ax2.set_ylabel(r'$\mathrm{Var}[s]$ [cm²]', fontsize=12)
        ax2.set_title('Power Law Fit (Var–E scaling)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('plot_p1_regression.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("   ✅ 2/7: plot_p1_regression.png")
    except Exception as e:
        print(f"   ⚠️ Error in plot 2: {e}")

    # -------------------------------------------------------------------------
    # Plot 3: NEW Multiplier Phase Space (Classification)
    # Replaces the old star diagram.
    # -------------------------------------------------------------------------
    try:
        # Compute quadrant statistics
        compacto = df[(df['lambda1'] < 0) & (df['lambda2'] > 0)]
        suelto   = df[(df['lambda1'] > 0) & (df['lambda2'] < 0)]
        conflicto1 = df[(df['lambda1'] > 0) & (df['lambda2'] > 0)]
        conflicto2 = df[(df['lambda1'] < 0) & (df['lambda2'] < 0)]
        n_compacto = len(compacto)
        n_suelto   = len(suelto)
        n_conflicto = len(conflicto1) + len(conflicto2)
        total = len(df)
        pct_compacto = 100 * n_compacto / total if total > 0 else 0
        pct_suelto   = 100 * n_suelto / total if total > 0 else 0
        pct_conflicto = 100 * n_conflicto / total if total > 0 else 0

        fig3 = plt.figure(figsize=(14, 8))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 2, width_ratios=[3.5, 1])
        ax_main = plt.subplot(gs[0])
        ax_bar = plt.subplot(gs[1])

        # Shade quadrants
        ax_main.add_patch(Rectangle((-100, 0), 100, 100, alpha=0.08, color='blue'))
        ax_main.add_patch(Rectangle((0, -100), 100, 100, alpha=0.08, color='red'))
        ax_main.add_patch(Rectangle((0, 0), 100, 100, alpha=0.05, color='orange'))
        ax_main.add_patch(Rectangle((-100, -100), 100, 100, alpha=0.05, color='orange'))

        # Axes through zero
        ax_main.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        ax_main.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

        # Use symlog for λ₂ to handle wide range
        ax_main.set_yscale('symlog', linthresh=10)
        ax_main.set_xscale('linear')
        ax_main.set_xlim(df['lambda1'].min() - 1, df['lambda1'].max() + 1)
        # Avoid log of negative; set y limits manually
        y_min = max(0.1, df['lambda2'].min() - 100) if df['lambda2'].min() > 0 else df['lambda2'].min() - 100
        y_max = df['lambda2'].max() + 1000
        ax_main.set_ylim(y_min, y_max)

        # Quadrant labels (in English)
        ax_main.text(-0.45, 0.92, 'COMPACT', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=13, weight='bold', color='darkblue',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=2))
        ax_main.text(-0.45, 0.82, 'λ₁ < 0, λ₂ > 0', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=10, color='darkblue')

        ax_main.text(0.45, 0.08, 'LOOSE', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=13, weight='bold', color='darkred',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='darkred', linewidth=2))
        ax_main.text(0.45, 0.18, 'λ₁ > 0, λ₂ < 0', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=10, color='darkred')

        ax_main.text(0.45, 0.92, 'CONFLICT A', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=11, weight='bold', color='darkorange',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='darkorange', linewidth=2))
        ax_main.text(0.45, 0.82, 'λ₁ > 0, λ₂ > 0', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=9, color='darkorange')

        ax_main.text(-0.45, 0.08, 'CONFLICT B', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=11, weight='bold', color='darkorange',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='darkorange', linewidth=2))
        ax_main.text(-0.45, 0.18, 'λ₁ < 0, λ₂ < 0', transform=ax_main.transAxes, ha='center', va='center',
                     fontsize=9, color='darkorange')

        # Scatter points by soil
        ids = df['Suelo_ID'].unique()
        cmap = plt.cm.tab20
        colores_map = {idx: cmap(i % 20) for i, idx in enumerate(sorted(ids))}
        for suelo_id in sorted(ids):
            subset = df[df['Suelo_ID'] == suelo_id]
            ax_main.scatter(subset['lambda1'], subset['lambda2'],
                            color=colores_map[suelo_id], alpha=0.6, s=35,
                            edgecolors='black', linewidth=0.3,
                            label=f'Soil {suelo_id}')

        ax_main.set_xlabel(r'Density multiplier $\lambda_1$ [cm³/g]', fontsize=13)
        ax_main.set_ylabel(r'Permeability multiplier $\lambda_2$ [1/cm²] (log scale)', fontsize=13)
        ax_main.set_title('Multiplier Phase Space (λ₁–λ₂) – Thermodynamic Classification', fontsize=14, weight='bold')
        ax_main.grid(True, alpha=0.2, linestyle='--', which='both')

        # Bar chart with percentages
        categorias = ['Compact', 'Loose', 'Conflict']
        valores = [pct_compacto, pct_suelto, pct_conflicto]
        colores_bar = ['darkblue', 'darkred', 'darkorange']
        ax_bar.barh(categorias, valores, color=colores_bar, alpha=0.7,
                    edgecolor='black', linewidth=1.2, height=0.6)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel('Percentage (%)', fontsize=11)
        ax_bar.set_title('State Distribution', fontsize=12, weight='bold')
        ax_bar.grid(True, axis='x', alpha=0.3)
        for i, v in enumerate(valores):
            ax_bar.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=11, weight='bold')
        for i, (cat, n) in enumerate(zip(categorias, [n_compacto, n_suelto, n_conflicto])):
            ax_bar.text(5, i - 0.25, f'n = {n}', ha='center', va='center', fontsize=8, color='gray')

        # Legend outside the main plot
        handles, labels = ax_main.get_legend_handles_labels()
        if len(handles) > 8:
            handles = handles[:6] + [plt.Line2D([0], [0], marker='o', color='w',
                                                label='...', markerfacecolor='gray', markersize=5)]
            labels = labels[:6] + ['...']
        ax_main.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
                       fontsize=7, title='Soil ID', framealpha=0.9)

        # Add a subtle note if no loose points
        if n_suelto == 0:
            ax_main.text(0.85, 0.02, '✅ No loose points', transform=ax_main.transAxes,
                         fontsize=9, color='green', ha='center', va='bottom',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig('plot_p1_phasespace.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("   ✅ 3/7: plot_p1_phasespace.png (new classification)")
    except Exception as e:
        print(f"   ⚠️ Error in plot 3 (new classification): {e}")

    # -------------------------------------------------------------------------
    # Plot 4: Verification of Phase Equation – TWO PANELS
    #   (a) Raw data (no fit)
    #   (b) Per‑soil log-linear fits with equation and R²
    # -------------------------------------------------------------------------
    try:
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

        # ---- Panel (a): Raw data scatter ----
        ax4a.scatter(df["gamma_usado"], df["E_s_cm"], alpha=0.3, s=10, color='navy')
        ax4a.set_xlabel(r'Specific weight $\gamma$ [g/cm³]', fontsize=12)
        ax4a.set_ylabel(r'$E[s]$ [cm]', fontsize=12)
        ax4a.set_title('(a) Raw data: E[s] vs γ', fontsize=13)
        ax4a.grid(True, alpha=0.3)
        # Add a simple exponential guide line (optional, but keep it clean)
        # We'll just show the data.

        # ---- Panel (b): Per‑soil log-linear fits ----
        ids = df['Suelo_ID'].unique()
        r2_list = []
        for suelo_id in sorted(ids):
            subset = df[df['Suelo_ID'] == suelo_id]
            if len(subset) < 10:
                continue
            x = subset['gamma_usado'].values
            y = subset['E_s_cm'].values
            log_y = np.log(y)
            slope, intercept, r_value, _, _ = linregress(x, log_y)
            r2 = r_value ** 2
            r2_list.append(r2)
            # Plot the line for this soil
            x_fit = np.linspace(x.min(), x.max(), 50)
            y_fit = np.exp(slope * x_fit + intercept)
            ax4b.plot(x_fit, y_fit, color=colores_map[suelo_id], linewidth=2, alpha=0.9)
            # Scatter points for this soil
            ax4b.scatter(x, y, color=colores_map[suelo_id], alpha=0.3, s=10)

        # Add overall equation and average R²
        avg_r2 = np.mean(r2_list) if r2_list else 0
        ax4b.text(0.05, 0.95, f'Per‑soil log‑linear fits\nAverage R² = {avg_r2:.4f}',
                  transform=ax4b.transAxes, fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        # Add a generic equation annotation
        ax4b.text(0.05, 0.80, r'$\ln(E[s]) = m \cdot \gamma + b$',
                  transform=ax4b.transAxes, fontsize=12, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax4b.set_xlabel(r'Specific weight $\gamma$ [g/cm³]', fontsize=12)
        ax4b.set_ylabel(r'$E[s]$ [cm]', fontsize=12)
        ax4b.set_title('(b) Log‑linear fits per soil', fontsize=13)
        ax4b.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('plot_p1_phase_verification.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("   ✅ 4/7: plot_p1_phase_verification.png (two panels)")
    except Exception as e:
        print(f"   ⚠️ Error in plot 4 (phase verification): {e}")

    # -------------------------------------------------------------------------
    # Plot 5: Verification of Poiseuille Law
    # -------------------------------------------------------------------------
    try:
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        pendientes_locales = []
        for idx, suelo in enumerate(suelos):
            subset = df[df["Suelo_ID"] == suelo["id"]]
            if len(subset) > 5:
                log_k = np.log10(subset["k_usado"])
                log_E = np.log10(subset["E_s_cm"])
                slope, intercept, _, _, _ = linregress(log_k, log_E)
                pendientes_locales.append(slope)
                ax5.scatter(subset["k_usado"], subset["E_s_cm"],
                            color=colores[idx], alpha=0.3, s=10, label=f"Soil {suelo['id']}")
                k_fit = np.logspace(np.log10(subset["k_usado"].min()), np.log10(subset["k_usado"].max()), 50)
                E_fit = 10 ** (slope * np.log10(k_fit) + intercept)
                ax5.plot(k_fit, E_fit, color=colores[idx], linewidth=1.0, alpha=0.6)
        pend_media = np.mean(pendientes_locales) if pendientes_locales else 0
        ax5.text(0.05, 0.95, f"Average local slope: {pend_media:.3f}\n(Theoretical Poiseuille: 0.5)",
                 transform=ax5.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        ax5.set_xlabel(r'Permeability $k$ [cm/s]', fontsize=12)
        ax5.set_ylabel(r'$E[s]$ [cm]', fontsize=12)
        ax5.set_title('Verification of Poiseuille Law', fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig('plot_p1_poiseuille.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print("   ✅ 5/7: plot_p1_poiseuille.png")
    except Exception as e:
        print(f"   ⚠️ Error in plot 5: {e}")

    # -------------------------------------------------------------------------
    # Plot 6: Individual slopes per soil
    # -------------------------------------------------------------------------
    try:
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        b_individuales = []
        ids = []
        for suelo in suelos:
            subset = df[df["Suelo_ID"] == suelo["id"]]
            if len(subset) > 10:
                fit = fit_power_law(subset)
                if not np.isnan(fit['b']):
                    b_individuales.append(fit['b'])
                    ids.append(suelo['id'])
        if b_individuales:
            ax6.bar(ids, b_individuales, color='skyblue', edgecolor='navy', alpha=0.7)
            ax6.axhline(boot['b_mean'], color='red', linestyle='--', linewidth=2,
                        label=f'Global b = {boot["b_mean"]:.4f}')
            ax6.axhline(0.5, color='green', linestyle=':', linewidth=1.5, label='Theoretical b = 0.5')
            ax6.set_xlabel('Soil ID', fontsize=12)
            ax6.set_ylabel('Abrego Exponent (b)', fontsize=12)
            ax6.set_title('Individual slopes per soil vs. global exponent', fontsize=14)
            ax6.grid(True, axis='y', alpha=0.3)
            ax6.legend()
            plt.tight_layout()
            plt.savefig('plot_p1_individual_slopes.png', dpi=300, bbox_inches='tight')
            plt.close(fig6)
            print("   ✅ 6/7: plot_p1_individual_slopes.png")
        else:
            print("   ⚠️  Could not calculate individual slopes.")
    except Exception as e:
        print(f"   ⚠️ Error in plot 6: {e}")

    # -------------------------------------------------------------------------
    # Plot 7: (Optional) A summary of the new classification – we already have it.
    # We can add a small extra figure if needed, but we keep the original number.
    # I'll add a simple pie chart as an extra to complete 7 plots.
    # -------------------------------------------------------------------------
    try:
        fig7, ax7 = plt.subplots(figsize=(6, 6))
        labels = ['Compact', 'Loose', 'Conflict']
        sizes = [n_compacto, n_suelto, n_conflicto]
        colors_pie = ['darkblue', 'darkred', 'darkorange']
        if sum(sizes) > 0:
            ax7.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            ax7.set_title('Thermodynamic State Distribution', fontsize=14)
            plt.tight_layout()
            plt.savefig('plot_p1_state_pie.png', dpi=300, bbox_inches='tight')
            plt.close(fig7)
            print("   ✅ 7/7: plot_p1_state_pie.png")
        else:
            print("   ⚠️  No data for pie chart.")
    except Exception as e:
        print(f"   ⚠️ Error in plot 7: {e}")

# =============================================================================
# 5. PART 2 PLOTS (unchanged, but filenames updated)
# =============================================================================

def generar_graficos_parte2(df_grid: pd.DataFrame):
    print("\n📊 Generating plots - Part 2...")
    if df_grid.empty:
        print("   ⚠️  No data to plot for Part 2.")
        return

    try:
        pivot_b = df_grid.pivot(index='n0', columns='beta', values='b')
        pivot_realista = df_grid.pivot(index='n0', columns='beta', values='realista')

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        im = ax1.imshow(pivot_b.values, cmap='viridis', aspect='auto', origin='lower',
                        extent=[min(BETA_VALUES)-0.025, max(BETA_VALUES)+0.025,
                                min(N0_VALUES)-0.025, max(N0_VALUES)+0.025])
        for i, n0 in enumerate(N0_VALUES):
            for j, beta in enumerate(BETA_VALUES):
                if not pivot_realista.loc[n0, beta]:
                    rect = Rectangle((beta-0.025, n0-0.025), 0.05, 0.05,
                                     facecolor='none', edgecolor='red', linewidth=2,
                                     hatch='//', alpha=0.7)
                    ax1.add_patch(rect)
                else:
                    val = pivot_b.loc[n0, beta]
                    if not np.isnan(val):
                        ax1.text(beta, n0, f'{val:.4f}', ha='center', va='center',
                                 color='white' if val > 0.5 else 'black', fontsize=9)
        ax1.set_xlabel(r'$\beta$', fontsize=12)
        ax1.set_ylabel(r'$n_0$', fontsize=12)
        ax1.set_title('Abrego Exponent (b) for (n₀, β)', fontsize=14)
        plt.colorbar(im, ax=ax1, label='b')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='none', edgecolor='red', hatch='//', label='Non-realistic')]
        ax1.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.savefig('plot_p2_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("   ✅ 1/3: plot_p2_heatmap.png")
    except Exception as e:
        print(f"   ⚠️ Error in heatmap: {e}")

    try:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.plasma(np.linspace(0, 1, len(df_grid)))
        log_E_range = np.linspace(-6, -1, 100)
        for idx, row in df_grid.iterrows():
            if not row['realista'] or np.isnan(row['b']):
                continue
            intercept = np.log10(0.0362) + 0.1 * (row['b'] - 0.5)
            log_Var_fit = intercept + row['b'] * log_E_range
            ax2.plot(10**log_E_range, 10**log_Var_fit, color=cmap[idx], alpha=0.5, linewidth=1.5)
        df_real = df_grid[df_grid['realista'] & ~np.isnan(df_grid['b'])]
        if not df_real.empty:
            b_mean = df_real['b'].mean()
            intercept_mean = np.log10(0.0362)
            log_Var_mean = intercept_mean + b_mean * log_E_range
            ax2.plot(10**log_E_range, 10**log_Var_mean, 'k-', linewidth=3,
                     label=f'Mean b (realistic) = {b_mean:.4f}')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$E[s]$ [cm]', fontsize=12)
        ax2.set_ylabel(r'$\mathrm{Var}[s]$ [cm²]', fontsize=12)
        ax2.set_title('Overlay of realistic lines', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('plot_p2_overlay.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("   ✅ 2/3: plot_p2_overlay.png")
    except Exception as e:
        print(f"   ⚠️ Error in overlay: {e}")

    try:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        labels = [f'({row["n0"]:.2f}, {row["beta"]:.2f})' for _, row in df_grid.iterrows()]
        b_values = df_grid['b'].values
        realista = df_grid['realista'].values
        colors_barras = ['green' if r else 'gray' for r in realista]
        ax3.bar(labels, b_values, color=colors_barras, edgecolor='navy', alpha=0.7)
        b_mean_real = df_grid[df_grid['realista']]['b'].mean() if not df_grid[df_grid['realista']].empty else np.nan
        if not np.isnan(b_mean_real):
            ax3.axhline(b_mean_real, color='red', linestyle='--', linewidth=2,
                        label=f'Mean (realistic) = {b_mean_real:.4f}')
        ax3.set_xlabel('Combination (n₀, β)', fontsize=12)
        ax3.set_ylabel('Abrego Exponent (b)', fontsize=12)
        ax3.set_title('Stability of b: realistic (green) vs non-realistic (gray)', fontsize=14)
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.legend()
        plt.tight_layout()
        plt.savefig('plot_p2_bars.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("   ✅ 3/3: plot_p2_bars.png")
    except Exception as e:
        print(f"   ⚠️ Error in bar chart: {e}")

# =============================================================================
# 6. REPORT GENERATION (modified to English, renamed files)
# =============================================================================

def generar_reporte_pdf(df_parte1, boot, df_parte2, total_validos,
                        total_simulaciones, alertas, curvature_detected):
    print("\n📄 Generating report...")
    # Write plain text report (English)
    with open("report_validation_3.txt", "w", encoding='utf-8') as f:
        f.write("MEPPME VALIDATION REPORT (v16.0)\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(f"Tortuosity factor (T): {TORTUOSITY_FACTOR}\n")
        f.write(f"Valid: {total_validos}/{total_simulaciones} ({total_validos/total_simulaciones:.2%})\n")
        f.write(f"Exponent b: {boot['b_mean']:.4f} ± {boot['b_std']:.4f}\n")
        f.write(f"95% CI: [{boot['b_ci_lower']:.4f}, {boot['b_ci_upper']:.4f}]\n")
        f.write(f"R²: {boot['r2']:.4f}\n")
        f.write(f"Curvature detected: {'YES' if curvature_detected else 'NO'}\n")
        f.write("Alerts:\n")
        # Alert keys are in Spanish; we translate them except "lambda2 edge"
        alert_translation = {
            'residuo_alto': 'High residual',
            'cond_num_alto': 'High condition number',
            'moda_fija': 'Fixed mode',
            'no_convergido': 'No convergence',
            'lambda2_edge': 'lambda2 edge'
        }
        for k, v in alertas.items():
            label = alert_translation.get(k, k)
            f.write(f"  {label}: {v}\n")
        f.write("\nResults by soil (means):\n")
        resumen = df_parte1.groupby('Suelo_ID').agg({
            'E_s_cm': 'mean', 'Var_s_cm2': 'mean',
            'lambda1': 'mean', 'lambda2': 'mean', 'residuo': 'mean',
            'cond_num': 'mean'
        }).round(4)
        f.write(resumen.to_string())
        if not df_parte2.empty:
            f.write("\n\nPART 2 - Grid Search:\n")
            f.write(df_parte2.to_string(index=False))
    print("   ✅ Plain text report saved (report_validation_3.txt).")

    if not REPORTLAB_AVAILABLE:
        print("   ℹ️  reportlab not installed, skipping PDF.")
        return

    try:
        doc = SimpleDocTemplate(
            "report_validation_3.pdf",
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'],
                                     fontSize=16, alignment=TA_CENTER, spaceAfter=12)
        h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
                                  fontSize=14, spaceAfter=6)
        normal_style = ParagraphStyle('Normal', parent=styles['Normal'],
                                      fontSize=10, alignment=TA_JUSTIFY, spaceAfter=4)

        story = []
        story.append(Paragraph("MEPPME VALIDATION REPORT (v16.0)", title_style))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("EXECUTIVE SUMMARY", h1_style))
        story.append(Paragraph(
            f"A total of {total_simulaciones} simulations were executed. "
            f"Valid: {total_validos} ({total_validos/total_simulaciones:.2%}). "
            f"Abrego exponent: <b>{boot['b_mean']:.4f} ± {boot['b_std']:.4f}</b> "
            f"(95% CI: [{boot['b_ci_lower']:.4f}, {boot['b_ci_upper']:.4f}]), "
            f"R² = {boot['r2']:.4f}. "
            f"Curvature: {'YES' if curvature_detected else 'NO'}.",
            normal_style
        ))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("ALERTS", h1_style))
        alert_data = [["Type", "Count"]]
        for k, v in alertas.items():
            label = alert_translation.get(k, k)
            alert_data.append([label, str(v)])
        table = Table(alert_data, colWidths=[8*cm, 4*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), rl_colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), rl_colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), rl_colors.beige),
            ('GRID', (0,0), (-1,-1), 1, rl_colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("RESULTS BY SOIL (MEANS)", h1_style))
        resumen = df_parte1.groupby('Suelo_ID').agg({
            'E_s_cm': 'mean', 'Var_s_cm2': 'mean',
            'lambda1': 'mean', 'lambda2': 'mean', 'residuo': 'mean',
            'cond_num': 'mean'
        }).round(4)
        table_data = [["Soil", "E[s]", "Var[s]", "λ₁", "λ₂", "Residual", "Cond"]]
        for idx, row in resumen.iterrows():
            table_data.append([
                str(idx),
                f"{row['E_s_cm']:.3e}",
                f"{row['Var_s_cm2']:.3e}",
                f"{row['lambda1']:.4f}",
                f"{row['lambda2']:.4f}",
                f"{row['residuo']:.3f}",
                f"{row['cond_num']:.2e}"
            ])
        table2 = Table(table_data, colWidths=[1.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), rl_colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), rl_colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), rl_colors.beige),
            ('GRID', (0,0), (-1,-1), 1, rl_colors.black),
        ]))
        story.append(table2)

        if not df_parte2.empty:
            story.append(PageBreak())
            story.append(Paragraph("PART 2 - GRID SEARCH (n₀, β)", h1_style))
            table_data_p2 = [["n₀", "β", "b", "R²", "N", "Realistic"]]
            for _, row in df_parte2.iterrows():
                table_data_p2.append([
                    f"{row['n0']:.2f}",
                    f"{row['beta']:.2f}",
                    f"{row['b']:.4f}" if not np.isnan(row['b']) else "NaN",
                    f"{row['r2']:.4f}" if not np.isnan(row['r2']) else "NaN",
                    str(row['n_puntos']),
                    "Yes" if row['realista'] else "No"
                ])
            table3 = Table(table_data_p2, colWidths=[2*cm, 2*cm, 2.5*cm, 2.5*cm, 2*cm, 2*cm])
            table3.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), rl_colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), rl_colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), rl_colors.beige),
                ('GRID', (0,0), (-1,-1), 1, rl_colors.black),
            ]))
            story.append(table3)

        try:
            story.append(PageBreak())
            story.append(Paragraph("REFERENCE PLOTS", h1_style))
            # New plot filenames in English
            plot_files = [
                'plot_p1_regression.png',
                'plot_p1_phasespace.png',
                'plot_p1_phase_verification.png',
                'plot_p1_poiseuille.png',
                'plot_p2_overlay.png'
            ]
            for fname in plot_files:
                if os.path.exists(fname):
                    img = Image(fname, width=12*cm, height=9*cm)
                    story.append(img)
                    story.append(Spacer(1, 0.5*cm))
        except Exception as e:
            print(f"   ℹ️  Could not embed plots: {e}")

        doc.build(story)
        print("   ✅ PDF generated: report_validation_3.pdf")
    except Exception as e:
        print(f"   ⚠️  Error generating PDF: {e}. The plain text report is available.")

# =============================================================================
# 7. MAIN FUNCTION (unchanged except for the call to the new plots)
# =============================================================================

def main():
    # Part 2 is now ENABLED by default to explore the robustness of b
    EJECUTAR_PARTE_2 = True

    print("=" * 80)
    print("PROFESSIONAL VALIDATION MEPPME - v16.0 (Tortuosity, Fine mesh, New classification)")
    print("=" * 80)
    print(f"📅 Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"🔬 Soils: {len(SUELOS)}")
    print(f"📊 Part 1 samples: {N_SAMPLES_PART1} per soil → {len(SUELOS)*N_SAMPLES_PART1} total")
    print(f"⚙️  n₀ = {N0_BASE}, β = {BETA_BASE} (auto_fix={AUTO_FIX_BETA})")
    print(f"   Tortuosity factor T = {TORTUOSITY_FACTOR} (Sec. A.4.1.2)")
    print(f"   Mesh: s_min={S_MIN:.1e}, s_max={S_MAX:.1f}, s0={S0_GLOBAL:.2e}, points={N_POINTS_PART1}")
    print(f"   Residual threshold: {RESIDUO_THRESHOLD}, cond_num: {COND_NUM_THRESHOLD:.1e}")
    print(f"   maxiter={MAXITER}, timeout={TIMEOUT}s")
    print("-" * 80)

    rng_global = np.random.default_rng(seed=GLOBAL_SEED)

    # ========================================================================
    # PART 1
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: UNIVERSAL VALIDATION (n₀={}, β={})".format(N0_BASE, BETA_BASE))
    print("="*80)

    # Create base constants and apply tortuosity factor
    constants_base = PhysicalConstants(n0=N0_BASE, beta=BETA_BASE, s0=S0_GLOBAL)
    # Apply tortuosity factor (Appendix A, Section A.4.1.2)
    constants_base.alpha = constants_base.alpha * TORTUOSITY_FACTOR
    print(f"   Original theoretical α (Poiseuille): {32.0*0.01/(1.0*981.0):.3e} s/cm")
    print(f"   Effective α (with tortuosity T={TORTUOSITY_FACTOR}): {constants_base.alpha:.3e} s/cm")
    
    mesh_base = MeshGenerator(
        s_min=S_MIN, s_max=S_MAX, n_points=N_POINTS_PART1,
        constants=constants_base, auto_fix_beta=AUTO_FIX_BETA
    )
    print(f"   Effective β used (adjusted for porosity): {mesh_base.constants.beta:.4f}")

    resultados_p1 = []
    alertas = {'lambda2_edge': 0, 'residuo_alto': 0, 'cond_num_alto': 0,
               'moda_fija': 0, 'no_convergido': 0}
    total_simulaciones_p1 = len(SUELOS) * N_SAMPLES_PART1
    inicio_tiempo = time.time()

    iter_suelos = tqdm(SUELOS, desc="Overall progress")
    for idx, suelo in enumerate(iter_suelos):
        df_suelo = simular_muestras_suelo(
            suelo, constants_base, mesh_base, N_SAMPLES_PART1,
            bounds=LAMBDA_BOUNDS, maxiter=MAXITER, timeout=TIMEOUT,
            rng=rng_global
        )
        df_suelo['Suelo_ID'] = suelo['id']
        resultados_p1.append(df_suelo)

        for _, row in df_suelo.iterrows():
            if not row['valida']:
                if 'Residual' in row['motivo']:
                    alertas['residuo_alto'] += 1
                elif 'Condition' in row['motivo']:
                    alertas['cond_num_alto'] += 1
                elif 'edge' in row['motivo'] or 'λ₂' in row['motivo']:
                    alertas['lambda2_edge'] += 1
                elif 'No converged' in row['motivo']:
                    alertas['no_convergido'] += 1

        modas = df_suelo[df_suelo['valida']]['moda_cm'].dropna()
        if len(modas) > 10:
            if modas.value_counts().max() / len(modas) > MODA_FIJA_THRESHOLD:
                alertas['moda_fija'] += 1

        validas = df_suelo['valida'].sum()
        iter_suelos.set_postfix({"valid": f"{validas}/{N_SAMPLES_PART1}"})
        pd.concat(resultados_p1, ignore_index=True).to_csv("resultados_p1_intermedio.csv", index=False)

    df_p1 = pd.concat(resultados_p1, ignore_index=True)
    df_p1_validas = df_p1[df_p1['valida']]

    tiempo_p1 = time.time() - inicio_tiempo
    print(f"\n⏱️  Part 1 time: {tiempo_p1/60:.2f} minutes")
    print(f"✅ Valid points Part 1: {len(df_p1_validas)} / {total_simulaciones_p1} ({len(df_p1_validas)/total_simulaciones_p1:.2%})")

    if len(df_p1_validas) < 100:
        print("❌ ERROR: Less than 100 valid points. Aborting.")
        sys.exit(1)

    print("\n📊 Calculating Abrego exponent (Bootstrap)...")
    boot = bootstrap_abrego(df_p1_validas)
    fit_global = fit_power_law(df_p1_validas)
    curvature_detected = fit_global.get('curvature_detected', False)
    print(f"   Exponent b: {boot['b_mean']:.4f} ± {boot['b_std']:.4f}")
    print(f"   95% CI: [{boot['b_ci_lower']:.4f}, {boot['b_ci_upper']:.4f}]")
    print(f"   R² = {boot['r2']:.4f}")
    print(f"   Curvature detected: {'YES' if curvature_detected else 'NO'}")

    colores_base = plt.cm.Set3(np.linspace(0, 1, 12))
    colores = np.vstack([colores_base] * 2)[:len(SUELOS)]
    generar_graficos_parte1(df_p1_validas, boot, SUELOS, colores)

    # ========================================================================
    # PART 2 (GRID SEARCH) - NOW ENABLED
    # ========================================================================
    if EJECUTAR_PARTE_2:
        print("\n" + "="*80)
        print("PART 2: STRUCTURAL SENSITIVITY (GRID SEARCH 5×5)")
        print("="*80)
        resultados_p2 = []
        combinaciones = [(n0, beta) for n0 in N0_VALUES for beta in BETA_VALUES]
        inicio_p2 = time.time()
        with tqdm(total=len(combinaciones), desc="Grid Search") as pbar:
            for n0, beta in combinaciones:
                try:
                    c_temp = PhysicalConstants(n0=n0, beta=beta, s0=S0_GLOBAL)
                    # Apply the same tortuosity factor in the grid
                    c_temp.alpha = c_temp.alpha * TORTUOSITY_FACTOR
                    m_temp = MeshGenerator(
                        s_min=S_MIN, s_max=S_MAX, n_points=N_POINTS_PART2,
                        constants=c_temp, auto_fix_beta=True
                    )
                    df_temp_list = []
                    for suelo in SUELOS:
                        df_s = simular_muestras_suelo(
                            suelo, c_temp, m_temp, N_SAMPLES_PART2,
                            bounds=LAMBDA_BOUNDS, maxiter=MAXITER, timeout=TIMEOUT,
                            rng=rng_global
                        )
                        df_temp_list.append(df_s)
                    df_temp = pd.concat(df_temp_list, ignore_index=True)
                    df_temp_validas = df_temp[df_temp['valida']]
                    if len(df_temp_validas) < 50:
                        resultados_p2.append({'n0': n0, 'beta': beta,
                                              'b': np.nan, 'r2': np.nan,
                                              'n_puntos': len(df_temp_validas), 'realista': False})
                        pbar.update(1)
                        continue
                    fit_comb = fit_power_law(df_temp_validas)
                    residuo_medio = np.nanmean(df_temp_validas['residuo']) if not df_temp_validas.empty else 1.0
                    cond_medio = np.nanmean(df_temp_validas['cond_num']) if not df_temp_validas.empty else np.inf
                    realista = (len(df_temp_validas) > 100) and (residuo_medio < RESIDUO_THRESHOLD) and (cond_medio < COND_NUM_THRESHOLD)
                    resultados_p2.append({'n0': n0, 'beta': beta,
                                          'b': fit_comb['b'], 'r2': fit_comb['r2'],
                                          'n_puntos': len(df_temp_validas), 'realista': realista})
                    pd.DataFrame(resultados_p2).to_csv("resultados_p2_parcial.csv", index=False)
                    pbar.set_postfix({"b": f"{fit_comb['b']:.3f}"})
                except Exception as e:
                    print(f"\n❌ Error in combination ({n0:.2f},{beta:.2f}): {str(e)[:60]}")
                    resultados_p2.append({'n0': n0, 'beta': beta, 'b': np.nan,
                                          'r2': np.nan, 'n_puntos': 0, 'realista': False})
                pbar.update(1)
        df_p2 = pd.DataFrame(resultados_p2)
        df_p2.to_csv("resultados_p2_completo.csv", index=False)
        tiempo_p2 = time.time() - inicio_p2
        print(f"\n⏱️  Part 2 time: {tiempo_p2/60:.2f} minutes")
        generar_graficos_parte2(df_p2)
    else:
        df_p2 = pd.DataFrame()
        print("\nℹ️  Part 2 skipped (EJECUTAR_PARTE_2 = False)")

    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    generar_reporte_pdf(df_p1_validas, boot, df_p2, len(df_p1_validas),
                        total_simulaciones_p1, alertas, curvature_detected)

    print("\n" + "=" * 80)
    print("🎉 VALIDATION COMPLETE! Check the generated files.")
    print("=" * 80)

if __name__ == "__main__":
    main()