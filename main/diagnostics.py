from __future__ import annotations

import numpy as np
from scipy.linalg import eigvals, solve_continuous_lyapunov

from .utils import centered_running_mean, relative_error, symmetrize


def mass_parameter(m: float) -> float:
    return 1.0 + m


def shear_from_state(state: np.ndarray) -> np.ndarray:
    return state[..., 0] - state[..., 1]


def total_mode_from_state(state: np.ndarray, m: float) -> np.ndarray:
    M = mass_parameter(m)
    return (state[..., 0] + m * state[..., 1]) / M


def linear_shear_variance_theory(S: float, m: float, R: float) -> float:
    return R / (S * mass_parameter(m))


def total_mode_variance_growth_theory(m: float, R: float) -> float:
    M = mass_parameter(m)
    return 2.0 * R / M**2


def linear_damped_drift_matrix(
    S: float, m: float, lambda_a: float, lambda_o: float
) -> np.ndarray:
    return np.array(
        [[-(S * m + lambda_a), S * m], [S, -(S + lambda_o)]],
        dtype=float,
    )


def diffusion_matrix(R: float) -> np.ndarray:
    return np.array([[2.0 * R, 0.0], [0.0, 0.0]], dtype=float)


def lyapunov_covariance(
    S: float, m: float, R: float, lambda_a: float, lambda_o: float
) -> np.ndarray:
    A = linear_damped_drift_matrix(S, m, lambda_a, lambda_o)
    eig = eigvals(A)
    if np.max(np.real(eig)) >= 0.0:
        raise ValueError("Drift matrix is not Hurwitz; stationary covariance does not exist.")
    Sigma = solve_continuous_lyapunov(A, -diffusion_matrix(R))
    return symmetrize(Sigma)


def running_shear_variance(shear: np.ndarray) -> np.ndarray:
    return centered_running_mean(shear**2)


def ensemble_variance(values: np.ndarray) -> np.ndarray:
    return np.var(values, axis=1, ddof=1)


def estimate_linear_growth_rate(times: np.ndarray, values: np.ndarray, fit_start: float) -> float:
    mask = times >= fit_start
    coeffs = np.polyfit(times[mask], values[mask], deg=1)
    return coeffs[0]


def empirical_covariance(state: np.ndarray) -> np.ndarray:
    flat = state.reshape(-1, state.shape[-1])
    return np.cov(flat, rowvar=False, ddof=1)


def covariance_error_table(empirical: np.ndarray, theoretical: np.ndarray) -> list[dict[str, float | str]]:
    labels = [("aa", 0, 0), ("ao", 0, 1), ("oo", 1, 1)]
    rows = []
    for label, i, j in labels:
        emp = float(empirical[i, j])
        theo = float(theoretical[i, j])
        rows.append(
            {
                "diagnostic": f"cov_{label}",
                "empirical": emp,
                "theoretical": theo,
                "absolute_error": abs(emp - theo),
                "relative_error": relative_error(emp, theo),
            }
        )
    return rows


def cubic_moment_balance_lhs(S_tilde: float, m: float, shear: np.ndarray) -> float:
    return S_tilde * mass_parameter(m) * float(np.mean(np.abs(shear) ** 3))


def cubic_moment_balance_rhs(R: float) -> float:
    return R


def effective_eddy_friction(S_tilde: float, shear: np.ndarray) -> float:
    second = float(np.mean(shear**2))
    third = float(np.mean(np.abs(shear) ** 3))
    return S_tilde * third / max(second, 1e-12)


def recover_S_from_variance(shear_variance: float, R: float, M: float) -> float:
    return R / (M * shear_variance)


def recover_M_from_total_mode_growth(diffusion_slope: float, R: float) -> float:
    if diffusion_slope <= 0.0:
        return np.nan
    return np.sqrt(2.0 * R / diffusion_slope)


def recover_S_eddy_from_variance(shear_variance: float, R: float, M: float) -> float:
    return R / (M * shear_variance)


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    centered = series - np.mean(series)
    variance = np.var(centered)
    if variance <= 0.0:
        return np.ones(max_lag + 1)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 : corr.size // 2 + max_lag + 1]
    counts = np.arange(series.size, series.size - max_lag - 1, -1)
    return corr / (counts * variance)
