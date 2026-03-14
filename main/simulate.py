from __future__ import annotations

from typing import Callable

import numpy as np

from .models import (
    euler_maruyama_step,
    linear_damped_drift,
    linear_undamped_drift,
    ou_shear_step,
    quadratic_drag_drift,
)


DriftFunction = Callable[[np.ndarray], np.ndarray]


def _simulate_two_layer(
    drift_function: DriftFunction,
    *,
    R: float,
    dt: float,
    n_steps: int,
    n_ens: int,
    seed: int,
    initial_state: np.ndarray | None = None,
    store_every: int = 1,
) -> dict[str, np.ndarray]:
    if store_every < 1:
        raise ValueError("store_every must be >= 1")
    state = np.zeros((n_ens, 2), dtype=float) if initial_state is None else initial_state.copy()
    noise_strength = np.sqrt(2.0 * R)
    n_store = n_steps // store_every + 1
    data = np.empty((n_store, n_ens, 2), dtype=float)
    times = np.empty(n_store, dtype=float)
    data[0] = state
    times[0] = 0.0
    rng = np.random.default_rng(seed)
    store_index = 1
    for step in range(1, n_steps + 1):
        state = euler_maruyama_step(
            state, drift_function(state), dt=dt, rng=rng, noise_strength=noise_strength
        )
        if step % store_every == 0:
            data[store_index] = state
            times[store_index] = step * dt
            store_index += 1
    return {"t": times, "state": data}


def simulate_linear_undamped(
    *,
    S: float,
    m: float,
    R: float,
    dt: float,
    n_steps: int,
    n_ens: int,
    seed: int,
    initial_state: np.ndarray | None = None,
    store_every: int = 1,
) -> dict[str, np.ndarray]:
    return _simulate_two_layer(
        lambda state: linear_undamped_drift(state, S=S, m=m),
        R=R,
        dt=dt,
        n_steps=n_steps,
        n_ens=n_ens,
        seed=seed,
        initial_state=initial_state,
        store_every=store_every,
    )


def simulate_linear_damped(
    *,
    S: float,
    m: float,
    R: float,
    lambda_a: float,
    lambda_o: float,
    dt: float,
    n_steps: int,
    n_ens: int,
    seed: int,
    initial_state: np.ndarray | None = None,
    store_every: int = 1,
) -> dict[str, np.ndarray]:
    return _simulate_two_layer(
        lambda state: linear_damped_drift(
            state, S=S, m=m, lambda_a=lambda_a, lambda_o=lambda_o
        ),
        R=R,
        dt=dt,
        n_steps=n_steps,
        n_ens=n_ens,
        seed=seed,
        initial_state=initial_state,
        store_every=store_every,
    )


def simulate_quadratic_drag(
    *,
    S_tilde: float,
    m: float,
    R: float,
    dt: float,
    n_steps: int,
    n_ens: int,
    seed: int,
    initial_state: np.ndarray | None = None,
    store_every: int = 1,
) -> dict[str, np.ndarray]:
    return _simulate_two_layer(
        lambda state: quadratic_drag_drift(state, S_tilde=S_tilde, m=m),
        R=R,
        dt=dt,
        n_steps=n_steps,
        n_ens=n_ens,
        seed=seed,
        initial_state=initial_state,
        store_every=store_every,
    )


def simulate_ou_shear(
    *,
    drift_rate: float,
    R: float,
    dt: float,
    n_steps: int,
    n_ens: int,
    seed: int,
    store_every: int = 1,
) -> dict[str, np.ndarray]:
    if store_every < 1:
        raise ValueError("store_every must be >= 1")
    shear = np.zeros(n_ens, dtype=float)
    n_store = n_steps // store_every + 1
    data = np.empty((n_store, n_ens), dtype=float)
    times = np.empty(n_store, dtype=float)
    data[0] = shear
    times[0] = 0.0
    rng = np.random.default_rng(seed)
    store_index = 1
    for step in range(1, n_steps + 1):
        shear = ou_shear_step(shear, dt=dt, rng=rng, drift_rate=drift_rate, R=R)
        if step % store_every == 0:
            data[store_index] = shear
            times[store_index] = step * dt
            store_index += 1
    return {"t": times, "shear": data}
