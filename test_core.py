#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
test_core.py - Pruebas Unitarias para mepme_core.py (MEPPME v5.4)
================================================================================
AUTOR:      José Luis Abrego Salazar
EMPRESA:    ABNALITIC
AÑO:        2026
LICENCIA:   MIT

DESCRIPCIÓN:
    Suite de pruebas unitarias para el núcleo del modelo MEPPME.
    Versión definitiva con tolerancias realistas y pruebas corregidas.

    EJECUCIÓN:
        pytest test_core.py -v
================================================================================
"""

import pytest
import numpy as np

from mepme_core import (
    PhysicalConstants,
    MeshGenerator,
    porosity,
    gamma_s_func,
    partition_function,
    compute_moments,
    compute_hessian,
    compute_metrics,
    compute_sensitivities,
    run_inference,
    conditioning_explanation,
    interpretar_resultados,
    objective_J,
    gradient_J,
)

try:
    from mepme_core import monte_carlo_simulation
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def safe_constants():
    return PhysicalConstants(n0=0.40, beta=0.05, s0=1e-4)


@pytest.fixture
def default_constants():
    return PhysicalConstants(n0=0.40, beta=0.12, s0=1e-4)


@pytest.fixture
def narrow_mesh(safe_constants):
    return MeshGenerator(
        s_min=1e-5,
        s_max=1e-1,
        n_points=128,
        constants=safe_constants,
        auto_fix_beta=False,
    )


@pytest.fixture
def wide_mesh_fix(default_constants):
    mesh = MeshGenerator(
        s_min=1e-8,
        s_max=1.0,
        n_points=64,
        constants=default_constants,
        auto_fix_beta=True,
    )
    return mesh


@pytest.fixture
def lambda_zero():
    return np.array([0.0, 0.0])


@pytest.fixture
def lambda_sample():
    return np.array([-2.0, 3.0])


# =============================================================================
# PRUEBAS
# =============================================================================

class TestPhysicalConstants:
    def test_default_values(self):
        c = PhysicalConstants()
        assert c.gamma_s == 2.65
        assert c.gamma_w == 1.00
        assert c.mu == 0.01
        assert c.g == 981.0
        assert c.n0 == 0.40
        assert c.beta == 0.12
        assert c.s0 == 1e-4
        expected_alpha = 32.0 * 0.01 / (1.00 * 981.0)
        assert c.alpha == pytest.approx(expected_alpha, rel=1e-9)

    def test_invalid_n0(self):
        with pytest.raises(ValueError, match="n0 debe estar en"):
            PhysicalConstants(n0=1.2)
        with pytest.raises(ValueError, match="n0 debe estar en"):
            PhysicalConstants(n0=-0.1)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta debe ser no negativo"):
            PhysicalConstants(beta=-0.1)

    def test_invalid_s0(self):
        with pytest.raises(ValueError, match="s0 debe ser positivo"):
            PhysicalConstants(s0=-1e-4)


class TestMeshGenerator:
    def test_valid_narrow_mesh(self, narrow_mesh):
        assert len(narrow_mesh.get_mesh()) == 128
        assert narrow_mesh.get_mesh()[0] == pytest.approx(1e-5, rel=1e-6)
        assert narrow_mesh.get_mesh()[-1] == pytest.approx(1e-1, rel=1e-6)
        assert np.all(narrow_mesh.get_weights() > 0)
        assert narrow_mesh.beta_adjusted is False

    def test_auto_fix_beta_activado(self, wide_mesh_fix):
        assert wide_mesh_fix.beta_adjusted is True
        assert wide_mesh_fix.original_beta == 0.12
        assert wide_mesh_fix.constants.beta < 0.12
        assert wide_mesh_fix.constants.beta > 0.0

    def test_auto_fix_beta_desactivado(self, default_constants):
        with pytest.raises(ValueError, match="Porosidad fuera de rango"):
            MeshGenerator(
                s_min=1e-8,
                s_max=1.0,
                n_points=64,
                constants=default_constants,
                auto_fix_beta=False,
            )

    def test_mesh_weights_sum(self, narrow_mesh):
        total_weight = np.sum(narrow_mesh.get_weights())
        expected = 1e-1 - 1e-5
        assert total_weight == pytest.approx(expected, rel=1e-6)

    def test_s0_out_of_range(self, default_constants):
        with pytest.raises(ValueError, match="s0 .* debe estar dentro"):
            MeshGenerator(
                s_min=1e-8,
                s_max=1e-4,
                n_points=64,
                constants=default_constants,
                auto_fix_beta=False,
            )
        with pytest.raises(ValueError, match="s0 .* debe estar dentro"):
            MeshGenerator(
                s_min=1e-4,
                s_max=1e-1,
                n_points=64,
                constants=default_constants,
                auto_fix_beta=False,
            )

    def test_invalid_parameters(self, default_constants):
        with pytest.raises(ValueError, match="s_min debe ser positivo"):
            MeshGenerator(s_min=0, s_max=1.0, constants=default_constants)
        with pytest.raises(ValueError, match="s_max .* mayor que s_min"):
            MeshGenerator(s_min=1.0, s_max=0.5, constants=default_constants)
        with pytest.raises(ValueError, match="n_points debe ser al menos"):
            MeshGenerator(s_min=1e-5, s_max=1e-1, n_points=8, constants=default_constants)


class TestPhysicalFunctions:
    def test_porosity_at_s0(self, safe_constants):
        n = porosity(safe_constants.s0, safe_constants)
        assert n == pytest.approx(safe_constants.n0, rel=1e-9)

    def test_porosity_vectorized(self, safe_constants):
        s_vals = np.array([1e-4, 1e-3, 1e-2])
        n_vals = porosity(s_vals, safe_constants)
        expected = safe_constants.n0 + safe_constants.beta * np.log(s_vals / safe_constants.s0)
        np.testing.assert_allclose(n_vals, expected, rtol=1e-9)

    def test_porosity_raises_for_negative_s(self, safe_constants):
        with pytest.raises(ValueError, match="Todos los valores de s deben ser > 0"):
            porosity(np.array([-1.0, 1e-4]), safe_constants)

    def test_gamma_s_func_monotonic(self, safe_constants, narrow_mesh):
        s = narrow_mesh.get_mesh()
        gamma = gamma_s_func(s, safe_constants)
        dgamma = np.gradient(gamma, s)
        assert np.all(dgamma < 0), "γ(s) no es estrictamente decreciente"

    def test_gamma_s_func_bounds(self, safe_constants, narrow_mesh):
        s = narrow_mesh.get_mesh()
        gamma = gamma_s_func(s, safe_constants)
        assert np.all(gamma >= safe_constants.gamma_w)
        assert np.all(gamma <= safe_constants.gamma_s)


class TestPartitionAndMoments:
    def test_partition_at_zero(self, safe_constants, narrow_mesh, lambda_zero):
        logZ = partition_function(lambda_zero, narrow_mesh, return_log=True)
        Z = np.exp(logZ)
        expected = narrow_mesh.get_mesh()[-1] - narrow_mesh.get_mesh()[0]
        assert Z == pytest.approx(expected, rel=1e-6)

    def test_partition_large_lambda(self, safe_constants, narrow_mesh):
        logZ = partition_function(np.array([-10.0, 10.0]), narrow_mesh, return_log=True)
        assert np.isfinite(logZ)

    def test_moments_at_zero(self, safe_constants, narrow_mesh, lambda_zero):
        mom = compute_moments(lambda_zero, narrow_mesh, safe_constants)
        s = narrow_mesh.get_mesh()
        w = narrow_mesh.get_weights()
        gamma = gamma_s_func(s, safe_constants)
        s2 = s ** 2

        prob_uniform = w / np.sum(w)
        np.testing.assert_allclose(mom['prob'], prob_uniform, rtol=1e-8)

        expected_E_gamma = np.sum(prob_uniform * gamma)
        assert mom['E_gamma'] == pytest.approx(expected_E_gamma, rel=1e-6)

        expected_E_s2 = np.sum(prob_uniform * s2)
        assert mom['E_s2'] == pytest.approx(expected_E_s2, rel=1e-6)

    def test_moments_normalization(self, safe_constants, narrow_mesh, lambda_sample):
        mom = compute_moments(lambda_sample, narrow_mesh, safe_constants)
        assert np.sum(mom['prob']) == pytest.approx(1.0, rel=1e-8)

    def test_partition_finite(self, safe_constants, narrow_mesh, lambda_sample):
        logZ = partition_function(lambda_sample, narrow_mesh, return_log=True)
        assert np.isfinite(logZ)


class TestHessian:
    def test_hessian_positive_semidefinite(self, safe_constants, narrow_mesh, lambda_sample):
        H, pinv, cond_num = compute_hessian(lambda_sample, narrow_mesh, safe_constants)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals >= -1e-12), f"Autovalores negativos: {eigvals}"

    def test_hessian_definite_positive_for_good_params(self, safe_constants, narrow_mesh):
        H, pinv, cond_num = compute_hessian(np.array([0.0, 0.0]), narrow_mesh, safe_constants)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals > 1e-12), f"Autovalores no positivos: {eigvals}"
        assert cond_num < 1e6, f"Número de condición muy alto: {cond_num}"

    def test_hessian_pinv_consistency(self, safe_constants, narrow_mesh, lambda_sample):
        H, pinv, _ = compute_hessian(lambda_sample, narrow_mesh, safe_constants)
        H_pinv_H = H @ pinv @ H
        np.testing.assert_allclose(H_pinv_H, H, rtol=1e-6, atol=1e-8)


class TestMetrics:
    def test_metrics_for_uniform_distribution(self, safe_constants, narrow_mesh):
        s = narrow_mesh.get_mesh()
        w = narrow_mesh.get_weights()
        prob_uniform = w / np.sum(w)

        metrics = compute_metrics(prob_uniform, narrow_mesh, safe_constants)
        expected_E = np.sum(prob_uniform * s)
        expected_Var = np.sum(prob_uniform * (s - expected_E) ** 2)

        assert metrics['E_s'] == pytest.approx(expected_E, rel=1e-6)
        assert metrics['Var_s'] == pytest.approx(expected_Var, rel=1e-6)
        assert metrics['mode'] == pytest.approx(s[np.argmax(prob_uniform)], rel=1e-6)
        assert 0.0 <= metrics['homogeneidad'] <= 1.0
        assert metrics['entropy'] > 0

    def test_metrics_for_delta_like_distribution(self, safe_constants, narrow_mesh):
        s = narrow_mesh.get_mesh()
        prob = np.zeros_like(s)
        idx = len(s) // 2
        prob[idx] = 1.0

        metrics = compute_metrics(prob, narrow_mesh, safe_constants)
        assert metrics['Var_s'] < 1e-12
        assert metrics['mode'] == pytest.approx(s[idx], rel=1e-6)
        assert metrics['homogeneidad'] > 0.9


class TestSensitivities:
    def test_sensitivities_at_zero(self, safe_constants, narrow_mesh, lambda_zero):
        sens = compute_sensitivities(lambda_zero, narrow_mesh, safe_constants)
        mom = compute_moments(lambda_zero, narrow_mesh, safe_constants)
        prob = mom['prob']
        s = narrow_mesh.get_mesh()
        gamma = gamma_s_func(s, safe_constants)
        s2 = s ** 2

        E_s = np.sum(prob * s)
        E_gamma = np.sum(prob * gamma)
        E_s2 = np.sum(prob * s2)

        Cov_s_gamma = np.sum(prob * (s - E_s) * (gamma - E_gamma))
        Cov_s_s2 = np.sum(prob * (s - E_s) * (s2 - E_s2))

        expected = -np.array([Cov_s_gamma, Cov_s_s2])
        np.testing.assert_allclose(sens, expected, rtol=1e-6)


class TestOptimization:
    def test_gradient_J_numerical(self, safe_constants, narrow_mesh):
        lambda0 = np.array([-2.0, 3.0])
        gamma_obj = 1.80
        k_obj = 1e-4
        eps = 1e-6
        cnt = {'count': 0}

        grad_analitico = gradient_J(lambda0, gamma_obj, k_obj, narrow_mesh, safe_constants)

        lambda_eps1 = np.array([lambda0[0] + eps, lambda0[1]])
        lambda_eps1_neg = np.array([lambda0[0] - eps, lambda0[1]])
        J_plus1 = objective_J(lambda_eps1, gamma_obj, k_obj, narrow_mesh, safe_constants, cnt)
        J_minus1 = objective_J(lambda_eps1_neg, gamma_obj, k_obj, narrow_mesh, safe_constants, cnt)
        grad_num1 = (J_plus1 - J_minus1) / (2 * eps)

        lambda_eps2 = np.array([lambda0[0], lambda0[1] + eps])
        lambda_eps2_neg = np.array([lambda0[0], lambda0[1] - eps])
        J_plus2 = objective_J(lambda_eps2, gamma_obj, k_obj, narrow_mesh, safe_constants, cnt)
        J_minus2 = objective_J(lambda_eps2_neg, gamma_obj, k_obj, narrow_mesh, safe_constants, cnt)
        grad_num2 = (J_plus2 - J_minus2) / (2 * eps)

        grad_numerico = np.array([grad_num1, grad_num2])
        np.testing.assert_allclose(grad_analitico, grad_numerico, rtol=5e-2, atol=1e-6)

    def test_fallback_bounds_consistency(self, safe_constants, narrow_mesh):
        res = run_inference(
            gamma_obj=1.80,
            k_obj=1e-4,
            mesh=narrow_mesh,
            constants=safe_constants,
            bounds=(-0.1, 0.1),
            use_fallbacks=True,
            maxiter=100,
        )
        assert isinstance(res, dict)
        if res.get('valida', False):
            assert -0.1 <= res['lambda'][0] <= 0.1
            assert -0.1 <= res['lambda'][1] <= 0.1


class TestInference:
    def test_run_inference_basic(self, safe_constants, narrow_mesh):
        gamma_obj = 1.80
        k_obj = 1e-4
        res = run_inference(gamma_obj, k_obj, narrow_mesh, safe_constants)
        assert res['success'] is True
        assert res['valida'] is True
        assert 'metrics' in res
        assert 'residual_norm' in res
        assert res['residual_norm'] < 0.05
        assert 'cond_num' in res

    def test_run_inference_no_fallbacks(self, safe_constants, narrow_mesh):
        """
        Prueba que cuando los bounds son demasiado estrechos y no se usan fallbacks,
        el optimizador se queda en la frontera y el residuo es alto.
        """
        res = run_inference(
            gamma_obj=1.80,
            k_obj=1e-4,
            mesh=narrow_mesh,
            constants=safe_constants,
            bounds=(-0.001, 0.001),
            use_fallbacks=False,
        )
        # Con el gradiente corregido, L-BFGS-B siempre encuentra un mínimo en la frontera
        assert res['success'] is True
        # El residuo es alto porque la solución no está dentro de los bounds
        assert res['residual_norm'] > 0.1

    def test_run_inference_invalid_inputs(self, safe_constants, narrow_mesh):
        with pytest.raises(ValueError, match="gamma_obj y k_obj deben ser positivos"):
            run_inference(gamma_obj=-1.0, k_obj=1e-4, mesh=narrow_mesh, constants=safe_constants)
        with pytest.raises(ValueError, match="gamma_obj y k_obj deben ser positivos"):
            run_inference(gamma_obj=1.80, k_obj=-1e-4, mesh=narrow_mesh, constants=safe_constants)

    def test_run_inference_with_auto_fix_mesh(self, default_constants):
        mesh = MeshGenerator(
            s_min=1e-8,
            s_max=1.0,
            n_points=128,
            constants=default_constants,
            auto_fix_beta=True,
        )
        res = run_inference(1.80, 1e-4, mesh, default_constants)
        assert isinstance(res, dict)
        if res['success']:
            assert res['valida'] is True


@pytest.mark.skipif(not MONTE_CARLO_AVAILABLE, reason="monte_carlo_simulation no está en mepme_core.py")
class TestMonteCarlo:
    def test_monte_carlo_basic(self, safe_constants, narrow_mesh):
        mc_result = monte_carlo_simulation(
            1.80, 1e-4, narrow_mesh, safe_constants,
            0.01, 1e-5, n_samples=20,
        )
        assert mc_result is not None
        assert 'lambda1' in mc_result
        assert 'lambda2' in mc_result
        assert 'E_s' in mc_result
        assert mc_result['n_valid'] > 0

    def test_monte_carlo_handles_negative_perturbations(self, safe_constants, narrow_mesh):
        mc_result = monte_carlo_simulation(
            1.80, 1e-4, narrow_mesh, safe_constants,
            0.5, 1e-3, n_samples=50,
        )
        assert mc_result is not None


class TestDiagnostics:
    def test_conditioning_explanation(self):
        assert "Bien condicionado" in conditioning_explanation(50)
        assert "Condicionamiento moderado" in conditioning_explanation(500)
        assert "Alta sensibilidad" in conditioning_explanation(5000)

    def test_interpretar_resultados(self):
        interp = interpretar_resultados(-1.0, 2.0, 1e-5, 1e-8)
        assert "Compacto" in interp['pos_estrella']
        interp = interpretar_resultados(1.0, -2.0, 1e-2, 1e-6)
        assert "Suelto" in interp['pos_estrella']
        interp = interpretar_resultados(0.01, 0.01, 1e-3, 1e-7)
        assert "Conflicto" in interp['pos_estrella']
        interp_neutro = interpretar_resultados(0.0, 0.0, 1e-3, 1e-7)
        assert "Neutro" in interp_neutro['pos_estrella']


class TestIntegration:
    def test_full_inference_pipeline(self, safe_constants, narrow_mesh):
        gamma_obj = 1.85
        k_obj = 1e-5

        res = run_inference(gamma_obj, k_obj, narrow_mesh, safe_constants)

        assert res['success'] is True
        assert res['valida'] is True
        assert abs(res['E_gamma'] - gamma_obj) < 1e-6
        assert abs(res['E_s2'] - safe_constants.alpha * k_obj) < 1e-3

        prob = res['prob']
        assert np.sum(prob) == pytest.approx(1.0, rel=1e-8)
        assert np.all(prob >= 0)

        metrics = res['metrics']
        assert metrics['E_s'] > 0
        assert metrics['Var_s'] >= 0
        assert 0 <= metrics['homogeneidad'] <= 1

        H = res['hessian']
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals > -1e-12)
        assert np.isfinite(res['cond_num'])


class TestRegression:
    def test_extreme_k_values(self, safe_constants, narrow_mesh):
        res = run_inference(1.80, 1e-12, narrow_mesh, safe_constants)
        assert isinstance(res, dict)
        if res['success'] and res.get('valida', False):
            assert 'metrics' in res

    def test_extreme_gamma_values(self, safe_constants, narrow_mesh):
        res = run_inference(2.50, 1e-5, narrow_mesh, safe_constants)
        assert isinstance(res, dict)
        if res['success'] and res.get('valida', False):
            assert 'metrics' in res

    def test_high_resolution_mesh_with_auto_fix(self, default_constants):
        mesh_high = MeshGenerator(
            s_min=1e-8,
            s_max=1.0,
            n_points=1024,
            constants=default_constants,
            auto_fix_beta=True,
        )
        assert mesh_high.beta_adjusted is True
        res = run_inference(1.80, 1e-4, mesh_high, default_constants)
        if res['success']:
            assert res['valida'] is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))