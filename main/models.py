from __future__ import annotations

import numpy as np


def linear_undamped_drift(state: np.ndarray, S: float, m: float) -> np.ndarray:
    shear = state[:, 0] - state[:, 1]
    drift = np.empty_like(state)
    drift[:, 0] = -S * m * shear
    drift[:, 1] = S * shear
    return drift


def linear_damped_drift(
    state: np.ndarray, S: float, m: float, lambda_a: float, lambda_o: float
) -> np.ndarray:
    shear = state[:, 0] - state[:, 1]
    drift = np.empty_like(state)
    drift[:, 0] = -S * m * shear - lambda_a * state[:, 0]
    drift[:, 1] = S * shear - lambda_o * state[:, 1]
    return drift


def quadratic_drag_drift(state: np.ndarray, S_tilde: float, m: float) -> np.ndarray:
    shear = state[:, 0] - state[:, 1]
    drag = np.abs(shear) * shear
    drift = np.empty_like(state)
    drift[:, 0] = -S_tilde * m * drag
    drift[:, 1] = S_tilde * drag
    return drift


def euler_maruyama_step(
    state: np.ndarray,
    drift: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    noise_strength: float,
) -> np.ndarray:
    increments = np.zeros_like(state)
    increments[:, 0] = noise_strength * np.sqrt(dt) * rng.standard_normal(state.shape[0])
    return state + drift * dt + increments


def ou_shear_step(
    shear: np.ndarray, dt: float, rng: np.random.Generator, drift_rate: float, R: float
) -> np.ndarray:
    noise = np.sqrt(2.0 * R * dt) * rng.standard_normal(shear.shape[0])
    return shear - drift_rate * shear * dt + noise
