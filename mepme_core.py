#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
mepme_core.py - Core of the MEPPME Model (Maximum Entropy Principle)
================================================================================
Version: 5.4 (Sign correction in gradient_J)
Author:  José Luis Abrego Salazar
Company: ABNALITIC
Year:    2026
License: MIT

DESCRIPTION:
    Module that implements the mathematical and numerical engine of the
    microstructural inference model based on the Maximum Entropy Principle
    (Appendix A of the thesis).

    CORRECTION v5.4:
        - gradient_J now returns (γ_obj - E[γ], α k_obj - E[s²]),
          which is the correct gradient of J(λ) according to (A.4.5).

DEPENDENCIES:
    numpy, scipy, typing, logging, warnings, time
================================================================================
"""

import time
import logging
import warnings
from typing import Optional, Tuple, Dict, Any, Callable, Union
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("MEPPME.core")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# 1. PHYSICAL CONSTANTS AND STRUCTURAL PARAMETERS
# =============================================================================

class PhysicalConstants:
    """
    Container for physical constants and model parameters.

    References:
        - Porosity: Eq. (A.2.4): n(s) = n0 + β·ln(s/s0)
        - Specific weight: Eq. (A.2.7): γ(s) = γs·(1-n) + γw·n
        - Poiseuille constant: Eq. (A.4.3): α = 32μ / (γw·g)
    """

    def __init__(
        self,
        gamma_s: float = 2.65,
        gamma_w: float = 1.00,
        mu: float = 0.01,
        g: float = 981.0,
        n0: float = 0.40,
        beta: float = 0.12,
        s0: float = 1e-4,
        EPSILON: float = 1e-15,
    ):
        self.gamma_s = gamma_s
        self.gamma_w = gamma_w
        self.mu = mu
        self.g = g
        self.n0 = n0
        self.beta = beta
        self.s0 = s0
        self.EPSILON = EPSILON
        self.alpha = 32.0 * mu / (gamma_w * g)
        self._validate()

    def _validate(self) -> None:
        if self.gamma_s <= 0:
            raise ValueError(f"gamma_s must be positive, got {self.gamma_s}")
        if self.gamma_w <= 0:
            raise ValueError(f"gamma_w must be positive, got {self.gamma_w}")
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.g <= 0:
            raise ValueError(f"g must be positive, got {self.g}")
        if not (0.0 <= self.n0 <= 1.0):
            raise ValueError(f"n0 must be in [0,1], got {self.n0}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.s0 <= 0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if self.EPSILON <= 0:
            raise ValueError(f"EPSILON must be positive, got {self.EPSILON}")

    def __repr__(self) -> str:
        return (
            f"PhysicalConstants(gamma_s={self.gamma_s}, gamma_w={self.gamma_w}, "
            f"mu={self.mu}, g={self.g}, n0={self.n0}, beta={self.beta}, s0={self.s0})"
        )


# =============================================================================
# 2. MESH GENERATOR (WITH EXACT WEIGHTS)
# =============================================================================

class MeshGenerator:
    """
    Generates a logarithmic mesh with exact integration weights.

    The weights are calculated using the trapezoidal rule in the original variable s:
        w_0 = (s_1 - s_0) / 2
        w_i = (s_{i+1} - s_{i-1}) / 2
        w_{N-1} = (s_{N-1} - s_{N-2}) / 2

    This guarantees that ∫ ds = s_max - s_min exactly (within rounding error).
    """

    def __init__(
        self,
        s_min: float,
        s_max: float,
        n_points: int = 512,
        constants: Optional[PhysicalConstants] = None,
        auto_fix_beta: bool = True,
    ):
        if constants is None:
            constants = PhysicalConstants()
        self.constants = constants
        self.beta_adjusted = False
        self.original_beta = constants.beta

        if s_min <= 0:
            raise ValueError(f"s_min must be positive, got {s_min}")
        if s_max <= s_min:
            raise ValueError(f"s_max ({s_max}) must be greater than s_min ({s_min})")
        if n_points < 16:
            raise ValueError(f"n_points must be at least 16, got {n_points}")
        if not (s_min < constants.s0 < s_max):
            raise ValueError(
                f"s0 ({constants.s0}) must be within [s_min, s_max] = [{s_min}, {s_max}]"
            )

        self.s = np.unique(np.geomspace(s_min, s_max, n_points))

        # Exact integration weights (trapezoidal rule in s)
        self.weights = np.zeros_like(self.s)
        self.weights[0] = (self.s[1] - self.s[0]) / 2.0
        self.weights[-1] = (self.s[-1] - self.s[-2]) / 2.0
        for i in range(1, len(self.s) - 1):
            self.weights[i] = (self.s[i + 1] - self.s[i - 1]) / 2.0

        try:
            self._check_porosity_range()
        except ValueError as e:
            if auto_fix_beta:
                ln_min = np.log(s_min / constants.s0)
                ln_max = np.log(s_max / constants.s0)
                beta_max = min(
                    constants.n0 / (-ln_min),
                    (1.0 - constants.n0) / ln_max
                )
                if beta_max <= 0:
                    raise ValueError(
                        f"Cannot adjust β: n0={constants.n0} or mesh range "
                        f"makes a physical porosity impossible."
                    ) from e
                new_beta = beta_max * 0.95
                self.constants = PhysicalConstants(
                    gamma_s=constants.gamma_s,
                    gamma_w=constants.gamma_w,
                    mu=constants.mu,
                    g=constants.g,
                    n0=constants.n0,
                    beta=new_beta,
                    s0=constants.s0,
                    EPSILON=constants.EPSILON,
                )
                self.beta_adjusted = True
                self.original_beta = constants.beta
                logger.warning(
                    f"β={constants.beta:.4f} saturates porosity. "
                    f"Automatically adjusted to β={new_beta:.4f}."
                )
                self._check_porosity_range()
            else:
                raise e

    def _check_porosity_range(self) -> None:
        n_vals = porosity(self.s, self.constants)
        if np.any(n_vals < 0) or np.any(n_vals > 1):
            min_n = np.min(n_vals)
            max_n = np.max(n_vals)
            raise ValueError(
                f"Porosity out of range [0,1]: min={min_n:.4f}, max={max_n:.4f}. "
                f"Current β: {self.constants.beta:.4f}."
            )

    def get_mesh(self) -> np.ndarray:
        return self.s

    def get_weights(self) -> np.ndarray:
        return self.weights

    def __repr__(self) -> str:
        base = f"MeshGenerator(s_min={self.s[0]:.2e}, s_max={self.s[-1]:.2e}, n_points={len(self.s)})"
        if self.beta_adjusted:
            base += f" [β adjusted: {self.original_beta:.4f} → {self.constants.beta:.4f}]"
        return base


# =============================================================================
# 3. PHYSICAL FUNCTIONS
# =============================================================================

def porosity(s: Union[float, np.ndarray], constants: Optional[PhysicalConstants] = None) -> np.ndarray:
    c = constants if constants is not None else PhysicalConstants()
    s = np.asarray(s, dtype=float)
    if np.any(s <= 0):
        raise ValueError(f"All s values must be > 0, found s <= 0")
    ratio = np.clip(s / c.s0, c.EPSILON, 1e12)
    return c.n0 + c.beta * np.log(ratio)


def gamma_s_func(s: Union[float, np.ndarray], constants: Optional[PhysicalConstants] = None) -> np.ndarray:
    c = constants if constants is not None else PhysicalConstants()
    n = porosity(s, c)
    return c.gamma_s * (1 - n) + c.gamma_w * n


def _log_integrand(
    lambda_vec: np.ndarray,
    s: np.ndarray,
    weights: np.ndarray,
    constants: PhysicalConstants,
) -> np.ndarray:
    if len(lambda_vec) != 2:
        raise ValueError("lambda_vec must have length 2")
    l1, l2 = lambda_vec[0], lambda_vec[1]
    gamma = gamma_s_func(s, constants)
    s2 = np.clip(s ** 2, 0, 1e150)
    exponent = np.clip(l1 * gamma + l2 * s2, -700, 700)
    log_w = np.log(np.maximum(weights, constants.EPSILON))
    return -exponent + log_w


def partition_function(
    lambda_vec: np.ndarray,
    mesh: MeshGenerator,
    constants: Optional[PhysicalConstants] = None,
    return_log: bool = True,
) -> Union[float, np.ndarray]:
    c = constants if constants is not None else mesh.constants
    log_int = _log_integrand(lambda_vec, mesh.get_mesh(), mesh.get_weights(), c)
    logZ = logsumexp(log_int)
    if return_log:
        return logZ
    else:
        return np.exp(logZ) if logZ > -700 else 0.0


def compute_moments(
    lambda_vec: np.ndarray,
    mesh: MeshGenerator,
    constants: Optional[PhysicalConstants] = None,
) -> Dict[str, Any]:
    c = constants if constants is not None else mesh.constants
    s = mesh.get_mesh()
    w = mesh.get_weights()

    log_int = _log_integrand(lambda_vec, s, w, c)
    logZ = logsumexp(log_int)

    prob = np.exp(log_int - logZ)
    prob = prob / np.sum(prob)

    gamma = gamma_s_func(s, c)
    s2 = np.clip(s ** 2, 0, 1e150)

    return {
        'prob': prob,
        'E_gamma': np.sum(prob * gamma),
        'E_s2': np.sum(prob * s2),
        'gamma_vals': gamma,
        's2_vals': s2,
        'logZ': logZ,
    }


def compute_hessian(
    lambda_vec: np.ndarray,
    mesh: MeshGenerator,
    constants: Optional[PhysicalConstants] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    res = compute_moments(lambda_vec, mesh, constants)
    prob = res['prob']
    gamma = res['gamma_vals']
    s2 = res['s2_vals']
    Eg = res['E_gamma']
    Es2 = res['E_s2']

    Eg2 = np.sum(prob * gamma ** 2)
    Es4 = np.sum(prob * s2 ** 2)
    Egs2 = np.sum(prob * gamma * s2)

    H = np.array([
        [Eg2 - Eg ** 2, Egs2 - Eg * Es2],
        [Egs2 - Eg * Es2, Es4 - Es2 ** 2],
    ])
    H = (H + H.T) / 2.0
    pinv = np.linalg.pinv(H, rcond=1e-15 * max(H.shape))
    cond_num = np.linalg.cond(H)
    return H, pinv, cond_num


def compute_metrics(
    prob: np.ndarray,
    mesh: MeshGenerator,
    constants: Optional[PhysicalConstants] = None,
) -> Dict[str, float]:
    c = constants if constants is not None else mesh.constants
    prob = np.asarray(prob, dtype=float)
    if np.abs(np.sum(prob) - 1) > 1e-6:
        prob = prob / np.sum(prob)

    s = mesh.get_mesh()
    w = mesh.get_weights()

    E_s = np.sum(prob * s)
    E_s2 = np.sum(prob * s ** 2)
    Var_s = E_s2 - E_s ** 2
    if Var_s < 0 and Var_s > -1e-12:
        Var_s = 0.0
    if Var_s < 0:
        Var_s = 0.0

    mode = s[np.argmax(prob)]
    mask = prob > c.EPSILON
    entropy = -np.sum(prob[mask] * np.log(prob[mask]) * w[mask])

    rango = s[-1] - s[0]
    homogeneidad = 1.0 - np.sqrt(Var_s) / rango if rango > 0 else 0.0
    homogeneidad = np.clip(homogeneidad, 0.0, 1.0)

    return {
        'E_s': E_s,
        'Var_s': Var_s,
        'mode': mode,
        'entropy': entropy,
        'homogeneidad': homogeneidad,
    }


def compute_sensitivities(
    lambda_vec: np.ndarray,
    mesh: MeshGenerator,
    constants: Optional[PhysicalConstants] = None,
) -> np.ndarray:
    c = constants if constants is not None else mesh.constants
    res = compute_moments(lambda_vec, mesh, c)
    prob = res['prob']
    s = mesh.get_mesh()
    gamma = gamma_s_func(s, c)
    s2 = np.clip(s ** 2, 0, 1e150)

    E_s = np.sum(prob * s)
    E_gamma = np.sum(prob * gamma)
    E_s2 = np.sum(prob * s2)

    Cov_s_gamma = np.sum(prob * (s - E_s) * (gamma - E_gamma))
    Cov_s_s2 = np.sum(prob * (s - E_s) * (s2 - E_s2))

    return -np.array([Cov_s_gamma, Cov_s_s2])


# =============================================================================
# 4. OPTIMIZATION (WITH CORRECTED GRADIENT)
# =============================================================================

class TimeoutException(Exception):
    pass


def _timeout_cb(start_time: float, limit: float) -> Callable:
    def cb(xk):
        if time.time() - start_time > limit:
            raise TimeoutException(f"Execution time exceeded ({limit} s)")
    return cb


def objective_J(
    lambda_vec: np.ndarray,
    gamma_obj: float,
    k_obj: float,
    mesh: MeshGenerator,
    constants: PhysicalConstants,
    counter: Dict[str, int],
) -> float:
    counter['count'] += 1
    try:
        logZ = partition_function(lambda_vec, mesh, constants, return_log=True)
        J = logZ + lambda_vec[0] * gamma_obj + lambda_vec[1] * (constants.alpha * k_obj)
        return J if np.isfinite(J) else 1e10
    except Exception:
        return 1e10


def gradient_J(
    lambda_vec: np.ndarray,
    gamma_obj: float,
    k_obj: float,
    mesh: MeshGenerator,
    constants: PhysicalConstants,
) -> np.ndarray:
    """
    Gradient of J(λ) according to (A.4.5).

    ∇J = (γ_obj - E[γ], α k_obj - E[s²])

    NOTE: This is the correct form, since dJ/dλ1 = -E[γ] + γ_obj.
    """
    try:
        res = compute_moments(lambda_vec, mesh, constants)
        return np.array([
            gamma_obj - res['E_gamma'],
            constants.alpha * k_obj - res['E_s2']
        ])
    except Exception:
        return np.zeros_like(lambda_vec)


def optimize_lbfgsb(
    lambda0: np.ndarray,
    gamma_obj: float,
    k_obj: float,
    mesh: MeshGenerator,
    constants: PhysicalConstants,
    bounds: Tuple[float, float] = (-100.0, 100.0),
    maxiter: int = 1000,
    timeout: float = 60.0,
    ftol: float = 1e-9,
    gtol: float = 1e-8,
) -> Dict[str, Any]:
    if len(lambda0) != 2:
        raise ValueError("lambda0 must have length 2")
    cnt = {'count': 0}
    start = time.time()

    def obj(x):
        return objective_J(x, gamma_obj, k_obj, mesh, constants, cnt)

    def grad(x):
        return gradient_J(x, gamma_obj, k_obj, mesh, constants)

    try:
        res = minimize(
            obj,
            lambda0,
            method='L-BFGS-B',
            jac=grad,
            bounds=[bounds] * 2,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
            callback=_timeout_cb(start, timeout),
        )
        return {
            'success': res.success,
            'lambda': res.x,
            'J': res.fun,
            'nit': res.nit,
            'message': res.message,
            'evals': cnt['count'],
        }
    except TimeoutException as e:
        logger.warning(f"L-BFGS-B: {e}")
        return {
            'success': False,
            'lambda': lambda0,
            'J': 1e10,
            'message': str(e),
            'evals': cnt['count'],
        }
    except Exception as e:
        logger.error(f"L-BFGS-B error: {e}")
        return {
            'success': False,
            'lambda': lambda0,
            'J': 1e10,
            'message': str(e),
            'evals': cnt['count'],
        }


def optimize_fallback(
    lambda0: np.ndarray,
    gamma_obj: float,
    k_obj: float,
    mesh: MeshGenerator,
    constants: PhysicalConstants,
    bounds: Tuple[float, float] = (-100.0, 100.0),
) -> Dict[str, Any]:
    cnt = {'count': 0}

    def obj(x):
        return objective_J(x, gamma_obj, k_obj, mesh, constants, cnt)

    try:
        res = differential_evolution(
            obj,
            [bounds] * 2,
            maxiter=500,
            popsize=15,
            tol=1e-6,
            seed=42,
            workers=1,
        )
        if res.success:
            return {
                'success': True,
                'lambda': res.x,
                'J': res.fun,
                'nit': res.nit,
                'message': 'DE converged',
                'evals': cnt['count'],
            }
        best = res.x
    except Exception as e:
        logger.warning(f"Differential Evolution failed: {e}")
        best = lambda0

    try:
        res = minimize(
            obj,
            best,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6},
        )
        if res.success:
            return {
                'success': True,
                'lambda': res.x,
                'J': res.fun,
                'message': 'Nelder-Mead',
                'evals': cnt['count'],
            }
    except Exception as e:
        logger.warning(f"Nelder-Mead failed: {e}")

    return {
        'success': False,
        'lambda': best,
        'J': 1e10,
        'message': 'All fallbacks failed',
        'evals': cnt['count'],
    }


def run_inference(
    gamma_obj: float,
    k_obj: float,
    mesh: MeshGenerator,
    constants: Optional[PhysicalConstants] = None,
    lambda0: Optional[np.ndarray] = None,
    use_fallbacks: bool = True,
    bounds: Tuple[float, float] = (-100.0, 100.0),
    maxiter: int = 1000,
    timeout: float = 60.0,
    ftol: float = 1e-9,
    gtol: float = 1e-8,
) -> Dict[str, Any]:
    if constants is None:
        constants = PhysicalConstants()
    if gamma_obj <= 0 or k_obj <= 0:
        raise ValueError("gamma_obj and k_obj must be positive")
    if lambda0 is None:
        lambda0 = np.array([0.0, 0.0])
    lambda0 = np.asarray(lambda0, dtype=float).flatten()
    if len(lambda0) != 2:
        raise ValueError("lambda0 length 2")

    logger.info(f"Inference: γ={gamma_obj:.4f}, k={k_obj:.4e}")

    result = optimize_lbfgsb(
        lambda0, gamma_obj, k_obj, mesh, constants,
        bounds=bounds, maxiter=maxiter, timeout=timeout,
        ftol=ftol, gtol=gtol,
    )

    if not result['success'] and not use_fallbacks:
        result['valida'] = False
        result['mensaje'] = 'Main optimization failed and fallbacks disabled'
        return result

    if not result['success'] and use_fallbacks:
        logger.warning("L-BFGS-B failed. Activating fallbacks.")
        fb = optimize_fallback(lambda0, gamma_obj, k_obj, mesh, constants, bounds=bounds)
        if fb['success']:
            result = fb
            logger.info("Fallback converged.")
        else:
            result['valida'] = False
            result['mensaje'] = 'Fallback also failed'
            return result

    if not result['success']:
        result['valida'] = False
        result['mensaje'] = result.get('message', 'Optimization failed')
        return result

    lf = result['lambda']
    try:
        mom = compute_moments(lf, mesh, constants)
        result['E_gamma'] = mom['E_gamma']
        result['E_s2'] = mom['E_s2']
        result['prob'] = mom['prob']
        result['gamma_obj'] = gamma_obj
        result['k_obj'] = k_obj
        result['alpha'] = constants.alpha

        residual = np.array([
            mom['E_gamma'] - gamma_obj,
            mom['E_s2'] - constants.alpha * k_obj,
        ])
        result['residual'] = residual
        result['residual_norm'] = np.linalg.norm(residual)

        H, pinv, cond_num = compute_hessian(lf, mesh, constants)
        result['hessian'] = H
        result['hessian_pinv'] = pinv
        result['cond_num'] = cond_num

        result['metrics'] = compute_metrics(mom['prob'], mesh, constants)
        result['consistency'] = {
            'rel_gamma': np.abs(residual[0]) / (gamma_obj + 1e-15),
            'rel_k': np.abs(residual[1]) / (constants.alpha * k_obj + 1e-15),
        }
        result['dE_s_dlambda'] = compute_sensitivities(lf, mesh, constants)

        result['valida'] = True
        result['mensaje'] = 'OK'

    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        result['valida'] = False
        result['mensaje'] = f'Post-proc error: {str(e)[:50]}'
        result['error'] = str(e)

    return result


# =============================================================================
# 5. DIAGNOSTIC UTILITIES
# =============================================================================

def conditioning_explanation(cond_num: float) -> str:
    if cond_num < 100:
        return "✅ **Well-conditioned.** The model is stable. Trust the results."
    elif cond_num < 1000:
        return (
            "⚠️ **Moderate conditioning.** The solution is acceptable, but "
            "small errors in the data can affect the curve. "
            "Consider increasing mesh resolution or adjusting β."
        )
    else:
        return (
            "🔴 **High sensitivity (κ > 1000).** The model is fragile. "
            "Increase mesh points (≥1024) or adjust n0 and β to stabilize."
        )


def interpretar_resultados(lambda1: float, lambda2: float, E_s: float, Var_s: float) -> Dict[str, str]:
    if lambda1 < 0 and lambda2 > 0:
        pos_estrella = "🔴 Compact (λ₁<0, λ₂>0)"
        desc_estrella = "Compact soil, low permeability, small pores."
    elif lambda1 > 0 and lambda2 < 0:
        pos_estrella = "🔵 Loose (λ₁>0, λ₂<0)"
        desc_estrella = "Loose soil, high permeability, large pores."
    elif lambda1 < 0 and lambda2 < 0:
        pos_estrella = "🟡 Conflict (both negative)"
        desc_estrella = "Intermediate tendency with conflict (dual porosity)."
    elif lambda1 > 0 and lambda2 > 0:
        pos_estrella = "🟡 Conflict (both positive)"
        desc_estrella = "Intermediate tendency with conflict (poorly graded)."
    else:
        pos_estrella = "⚪ Neutral (near 0)"
        desc_estrella = "Neutral structure without clear tendency."

    if Var_s < 1e-8 and E_s < 1e-4:
        pos_momento = "📌 Very small and homogeneous pores"
        desc_momento = "Uniform and fine microstructure, low permeability."
    elif Var_s > 1e-6 and E_s > 1e-3:
        pos_momento = "📌 Large and heterogeneous pores"
        desc_momento = "Coarse and dispersed microstructure, good drainage."
    elif Var_s < 1e-8 and E_s > 1e-3:
        pos_momento = "📌 Large and homogeneous pores"
        desc_momento = "Uniform coarse microstructure, well-graded."
    elif Var_s > 1e-6 and E_s < 1e-4:
        pos_momento = "📌 Small but scattered pores"
        desc_momento = "Fine microstructure with microfractures."
    else:
        pos_momento = "📌 Intermediate zone"
        desc_momento = "Transitional microstructure."

    return {
        'pos_estrella': pos_estrella,
        'desc_estrella': desc_estrella,
        'pos_momento': pos_momento,
        'desc_momento': desc_momento,
    }


# =============================================================================
# 6. VERSION AND METADATA
# =============================================================================

__version__ = "5.4"
__author__ = "José Luis Abrego Salazar"
__email__ = "jlabrego@abnalitic.com"
__license__ = "MIT"
__all__ = [
    "PhysicalConstants",
    "MeshGenerator",
    "porosity",
    "gamma_s_func",
    "partition_function",
    "compute_moments",
    "compute_hessian",
    "compute_metrics",
    "compute_sensitivities",
    "run_inference",
    "conditioning_explanation",
    "interpretar_resultados",
    "__version__",
]

# End of mepme_core.py