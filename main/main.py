from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import numpy as np
import pandas as pd

from .diagnostics import (
    autocorrelation,
    covariance_error_table,
    cubic_moment_balance_lhs,
    cubic_moment_balance_rhs,
    effective_eddy_friction,
    empirical_covariance,
    ensemble_variance,
    estimate_linear_growth_rate,
    linear_shear_variance_theory,
    lyapunov_covariance,
    mass_parameter,
    recover_M_from_total_mode_growth,
    recover_S_eddy_from_variance,
    recover_S_from_variance,
    running_shear_variance,
    shear_from_state,
    total_mode_from_state,
    total_mode_variance_growth_theory,
)
from .figures import create_figure1, create_figure2, create_figure3, create_figure4, create_figure5
from .simulate import simulate_linear_damped, simulate_linear_undamped, simulate_ou_shear, simulate_quadratic_drag
from .utils import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, TABLES_DIR, configure_matplotlib, ensure_output_dirs, relative_error


@dataclass(frozen=True)
class LinearUndampedConfig:
    S: float = 0.5
    m: float = 2.0
    R: float = 1.0
    dt: float = 0.01
    n_steps: int = 12000
    burn_steps: int = 2000
    ensemble_size: int = 240
    seed_sample: int = 21
    seed_ensemble: int = 22


@dataclass(frozen=True)
class LinearDampedConfig:
    S: float = 0.5
    m: float = 2.0
    R: float = 1.0
    lambda_a: float = 0.2
    lambda_o: float = 0.1
    dt: float = 0.01
    n_steps: int = 14000
    burn_steps: int = 2500
    ensemble_size: int = 180
    seed_sample: int = 31
    seed_ensemble: int = 32


@dataclass(frozen=True)
class QuadraticConfig:
    S_tilde: float = 0.5
    m: float = 2.0
    R: float = 1.0
    dt: float = 0.005
    n_steps: int = 24000
    burn_steps: int = 4000
    ensemble_size: int = 120
    seed_sample: int = 41
    seed_ensemble: int = 42


def summarise_row(name: str, empirical: float, theoretical: float) -> dict[str, float | str]:
    return {
        "diagnostic": name,
        "empirical": empirical,
        "theoretical": theoretical,
        "absolute_error": abs(empirical - theoretical),
        "relative_error": relative_error(empirical, theoretical),
    }


def regression_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_squared)}


def run_linear_undamped_experiment(config: LinearUndampedConfig) -> tuple[dict, list[dict]]:
    sample = simulate_linear_undamped(
        S=config.S,
        m=config.m,
        R=config.R,
        dt=config.dt,
        n_steps=config.n_steps,
        n_ens=1,
        seed=config.seed_sample,
    )
    ensemble = simulate_linear_undamped(
        S=config.S,
        m=config.m,
        R=config.R,
        dt=config.dt,
        n_steps=config.n_steps,
        n_ens=config.ensemble_size,
        seed=config.seed_ensemble,
    )
    sample_state = sample["state"][:, 0, :]
    ensemble_state = ensemble["state"]

    M = mass_parameter(config.m)
    shear_theory = linear_shear_variance_theory(config.S, config.m, config.R)
    total_growth_theory = total_mode_variance_growth_theory(config.m, config.R)

    sample_shear = shear_from_state(sample_state)
    stationary_shear = sample_shear[config.burn_steps :]
    ensemble_stationary_shear = shear_from_state(ensemble_state[config.burn_steps :]).reshape(-1)
    running_var = running_shear_variance(stationary_shear)
    shear_empirical = float(np.mean(ensemble_stationary_shear**2))

    total_mode = total_mode_from_state(ensemble_state, config.m)
    total_variance = ensemble_variance(total_mode)
    total_growth_empirical = estimate_linear_growth_rate(
        ensemble["t"], total_variance, fit_start=sample["t"][config.burn_steps]
    )

    verification_rows = [
        summarise_row("linear_shear_variance", shear_empirical, shear_theory),
        summarise_row("total_mode_variance_growth", total_growth_empirical, total_growth_theory),
    ]

    payload = {
        "time_sample": sample["t"],
        "ua_sample": sample_state[:, 0],
        "uo_sample": sample_state[:, 1],
        "us_sample": sample_shear,
        "time_ensemble": ensemble["t"],
        "ut_variance": total_variance,
        "ut_variance_theory": total_growth_theory * ensemble["t"],
        "time_stationary": sample["t"][config.burn_steps :] - sample["t"][config.burn_steps],
        "running_var": running_var,
        "shear_variance_theory": shear_theory,
        "shear_variance_empirical": shear_empirical,
        "shear_variance_relerr": relative_error(shear_empirical, shear_theory),
        "total_growth_empirical": total_growth_empirical,
        "total_growth_relerr": relative_error(total_growth_empirical, total_growth_theory),
        "stationary_shear_hist": ensemble_stationary_shear,
        "boltzmann_std": float(np.sqrt(shear_theory)),
    }
    return payload, verification_rows


def run_linear_damped_experiment(config: LinearDampedConfig) -> tuple[dict, list[dict]]:
    sample = simulate_linear_damped(
        S=config.S,
        m=config.m,
        R=config.R,
        lambda_a=config.lambda_a,
        lambda_o=config.lambda_o,
        dt=config.dt,
        n_steps=config.n_steps,
        n_ens=1,
        seed=config.seed_sample,
    )
    ensemble = simulate_linear_damped(
        S=config.S,
        m=config.m,
        R=config.R,
        lambda_a=config.lambda_a,
        lambda_o=config.lambda_o,
        dt=config.dt,
        n_steps=config.n_steps,
        n_ens=config.ensemble_size,
        seed=config.seed_ensemble,
    )

    sample_state = sample["state"][:, 0, :]
    post_burn = ensemble["state"][config.burn_steps :]
    cov_theory = lyapunov_covariance(
        config.S, config.m, config.R, config.lambda_a, config.lambda_o
    )
    cov_emp = empirical_covariance(post_burn)

    running_cov_aa = np.array([np.mean(post_burn[:i, :, 0] ** 2) for i in range(50, post_burn.shape[0])])
    running_cov_ao = np.array(
        [np.mean(post_burn[:i, :, 0] * post_burn[:i, :, 1]) for i in range(50, post_burn.shape[0])]
    )
    running_cov_oo = np.array([np.mean(post_burn[:i, :, 1] ** 2) for i in range(50, post_burn.shape[0])])
    cov_times = ensemble["t"][config.burn_steps + 50 :] - ensemble["t"][config.burn_steps]

    verification_rows = covariance_error_table(cov_emp, cov_theory)

    scatter_state = post_burn[::20].reshape(-1, 2)
    payload = {
        "time": sample["t"],
        "ua": sample_state[:, 0],
        "uo": sample_state[:, 1],
        "cov_times": cov_times,
        "covariance_series": [
            (r"$\Sigma_{aa}$", running_cov_aa, cov_theory[0, 0], r"$\Sigma_{aa}$ (Lyapunov)"),
            (r"$\Sigma_{ao}$", running_cov_ao, cov_theory[0, 1], r"$\Sigma_{ao}$ (Lyapunov)"),
            (r"$\Sigma_{oo}$", running_cov_oo, cov_theory[1, 1], r"$\Sigma_{oo}$ (Lyapunov)"),
        ],
        "empirical_entries": [cov_emp[0, 0], cov_emp[0, 1], cov_emp[1, 1]],
        "theoretical_entries": [cov_theory[0, 0], cov_theory[0, 1], cov_theory[1, 1]],
        "covariance_mean_relerr": float(
            np.mean(
                [
                    relative_error(cov_emp[0, 0], cov_theory[0, 0]),
                    relative_error(cov_emp[0, 1], cov_theory[0, 1]),
                    relative_error(cov_emp[1, 1], cov_theory[1, 1]),
                ]
            )
        ),
        "scatter_state": scatter_state,
        "cov_theory": cov_theory,
    }
    return payload, verification_rows


def run_quadratic_experiment(config: QuadraticConfig) -> tuple[dict, list[dict]]:
    sample = simulate_quadratic_drag(
        S_tilde=config.S_tilde,
        m=config.m,
        R=config.R,
        dt=config.dt,
        n_steps=config.n_steps,
        n_ens=1,
        seed=config.seed_sample,
    )
    ensemble = simulate_quadratic_drag(
        S_tilde=config.S_tilde,
        m=config.m,
        R=config.R,
        dt=config.dt,
        n_steps=config.n_steps,
        n_ens=config.ensemble_size,
        seed=config.seed_ensemble,
    )
    sample_state = sample["state"][:, 0, :]
    sample_shear = shear_from_state(sample_state)
    post_burn_shear = shear_from_state(ensemble["state"][config.burn_steps :]).reshape(-1)

    lhs = cubic_moment_balance_lhs(config.S_tilde, config.m, post_burn_shear)
    rhs = cubic_moment_balance_rhs(config.R)
    S_eddy = effective_eddy_friction(config.S_tilde, post_burn_shear)

    max_lag = 300
    nonlinear_acf = autocorrelation(sample_shear[config.burn_steps :], max_lag=max_lag)
    lags = np.arange(max_lag + 1) * config.dt
    effective_acf = np.exp(-S_eddy * mass_parameter(config.m) * lags)

    sweep_params = [
        (0.2, 0.5),
        (0.2, 2.0),
        (0.5, 2.0),
        (0.8, 2.0),
        (1.0, 4.0),
    ]
    lhs_values = []
    rhs_values = []
    normalized_lhs_values = []
    normalized_rhs_values = []
    labels = []
    for idx, (S_tilde, m) in enumerate(sweep_params):
        sim = simulate_quadratic_drag(
            S_tilde=S_tilde,
            m=m,
            R=config.R,
            dt=config.dt,
            n_steps=12000,
            n_ens=80,
            seed=100 + idx,
        )
        sweep_shear = shear_from_state(sim["state"][2500:]).reshape(-1)
        sweep_lhs = cubic_moment_balance_lhs(S_tilde, m, sweep_shear)
        sweep_rhs = cubic_moment_balance_rhs(config.R)
        lhs_values.append(sweep_lhs)
        rhs_values.append(sweep_rhs)
        normalized_lhs_values.append(sweep_lhs / config.R)
        normalized_rhs_values.append(sweep_rhs / config.R)
        labels.append(rf"$\tilde S={S_tilde:.1f},\,m={m:.1f}$")

    verification_rows = [
        summarise_row("quadratic_cubic_moment_balance", lhs, rhs),
        {
            "diagnostic": "quadratic_effective_S_eddy",
            "empirical": S_eddy,
            "theoretical": np.nan,
            "absolute_error": np.nan,
            "relative_error": np.nan,
        },
    ]

    payload = {
        "time": sample["t"],
        "ua": sample_state[:, 0],
        "uo": sample_state[:, 1],
        "shear_hist": post_burn_shear,
        "shear_variance": float(np.mean(post_burn_shear**2)),
        "shear_abs3": float(np.mean(np.abs(post_burn_shear) ** 3)),
        "hist_mean": float(np.mean(post_burn_shear)),
        "hist_std": float(np.std(post_burn_shear, ddof=1)),
        "sweep_index": np.arange(len(sweep_params)),
        "lhs_values": lhs_values,
        "rhs_values": rhs_values,
        "normalized_lhs_values": normalized_lhs_values,
        "normalized_rhs_values": normalized_rhs_values,
        "sweep_labels": labels,
        "balance_empirical": lhs,
        "balance_theory": rhs,
        "balance_relerr": relative_error(lhs, rhs),
        "rhs_label": r"RHS: $1$",
        "acf_lags": lags,
        "acf_nonlinear": nonlinear_acf,
        "acf_effective": effective_acf,
        "S_eddy": S_eddy,
    }
    return payload, verification_rows


def run_parameter_recovery() -> tuple[dict, pd.DataFrame]:
    S_values = np.linspace(0.2, 1.0, 5)
    m_values = np.linspace(0.5, 4.0, 5)
    dt = 0.01
    n_steps = 9000
    burn_steps = 2000
    R = 1.0

    rows = []
    heatmap = np.empty((m_values.size, S_values.size))
    heatmap[:] = np.nan
    seed_counter = 200

    for j, m in enumerate(m_values):
        for i, S in enumerate(S_values):
            M = mass_parameter(m)
            sim = simulate_linear_undamped(
                S=S,
                m=m,
                R=R,
                dt=dt,
                n_steps=n_steps,
                n_ens=120,
                seed=seed_counter,
            )
            seed_counter += 1
            state = sim["state"]
            shear = shear_from_state(state[burn_steps:]).reshape(-1)
            total_mode = total_mode_from_state(state, m)
            total_variance = ensemble_variance(total_mode)
            growth = estimate_linear_growth_rate(sim["t"], total_variance, fit_start=sim["t"][burn_steps])
            S_obs = recover_S_from_variance(float(np.mean(shear**2)), R, M)
            M_obs = recover_M_from_total_mode_growth(growth, R)

            rows.append(
                {
                    "model": "linear",
                    "true_S": S,
                    "recovered_S": S_obs,
                    "relerr_S": relative_error(S_obs, S),
                    "true_M": M,
                    "recovered_M": M_obs,
                    "relerr_M": relative_error(M_obs, M),
                    "true_S_eddy": np.nan,
                    "recovered_S_eddy": np.nan,
                    "relerr_S_eddy": np.nan,
                    "m": m,
                }
            )
            heatmap[j, i] = 0.5 * (
                relative_error(S_obs, S) + relative_error(M_obs, M)
            )

    quad_pairs = [(S_tilde, m) for S_tilde in np.linspace(0.2, 1.0, 5) for m in (0.5, 2.0, 4.0)]
    for idx, (S_tilde, m) in enumerate(quad_pairs):
        M = mass_parameter(m)
        sim = simulate_quadratic_drag(
            S_tilde=S_tilde,
            m=m,
            R=R,
            dt=0.005,
            n_steps=15000,
            n_ens=100,
            seed=400 + idx,
        )
        shear = shear_from_state(sim["state"][3000:]).reshape(-1)
        S_eddy_true = effective_eddy_friction(S_tilde, shear)
        variance = float(np.mean(shear**2))
        S_eddy_obs = recover_S_eddy_from_variance(variance, R, M)
        rows.append(
            {
                "model": "quadratic",
                "true_S": np.nan,
                "recovered_S": np.nan,
                "relerr_S": np.nan,
                "true_M": M,
                "recovered_M": np.nan,
                "relerr_M": np.nan,
                "true_S_eddy": S_eddy_true,
                "recovered_S_eddy": S_eddy_obs,
                "relerr_S_eddy": relative_error(S_eddy_obs, S_eddy_true),
                "m": m,
            }
        )

    results = pd.DataFrame(rows)
    true_S = results.loc[results["model"] == "linear", "true_S"].to_numpy()
    recovered_S = results.loc[results["model"] == "linear", "recovered_S"].to_numpy()
    true_M = results.loc[results["model"] == "linear", "true_M"].to_numpy()
    recovered_M = results.loc[results["model"] == "linear", "recovered_M"].to_numpy()
    payload = {
        "true_S": true_S,
        "recovered_S": recovered_S,
        "true_M": true_M,
        "recovered_M": recovered_M,
        "relerr_S": results.loc[results["model"] == "linear", "relerr_S"].dropna().to_numpy(),
        "relerr_M": results.loc[results["model"] == "linear", "relerr_M"].dropna().to_numpy(),
        "relerr_S_eddy": results.loc[results["model"] == "quadratic", "relerr_S_eddy"].dropna().to_numpy(),
        "heatmap": heatmap,
        "heatmap_extent": [S_values.min(), S_values.max(), m_values.min(), m_values.max()],
        "fit_S": regression_metrics(true_S, recovered_S),
        "fit_M": regression_metrics(true_M, recovered_M),
    }
    return payload, results


def run_convergence_checks() -> list[dict]:
    rows = []
    base = LinearUndampedConfig()
    for dt in (0.02, 0.01, 0.005):
        n_steps = int(80.0 / dt)
        burn_steps = int(20.0 / dt)
        sim = simulate_linear_undamped(
            S=base.S,
            m=base.m,
            R=base.R,
            dt=dt,
            n_steps=n_steps,
            n_ens=100,
            seed=int(5000 * dt) + 9,
        )
        shear = shear_from_state(sim["state"][burn_steps:]).reshape(-1)
        empirical = float(np.mean(shear**2))
        theory = linear_shear_variance_theory(base.S, base.m, base.R)
        rows.append(
            {
                "diagnostic": "dt_convergence_linear_shear",
                "dt": dt,
                "empirical": empirical,
                "theoretical": theory,
                "absolute_error": abs(empirical - theory),
                "relative_error": relative_error(empirical, theory),
            }
        )
    return rows


def print_console_summary(summary: pd.DataFrame) -> None:
    print("Numerical verification summary")
    print("-" * 78)
    subset = summary.loc[:, ["diagnostic", "empirical", "theoretical", "relative_error"]]
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.6f}".format):
        print(subset.to_string(index=False))
    print("-" * 78)
    if "dt" in summary.columns:
        conv = summary[summary["diagnostic"] == "dt_convergence_linear_shear"]
        if not conv.empty:
            print("Selected dt convergence check:")
            with pd.option_context("display.float_format", "{:.6f}".format):
                print(conv.loc[:, ["dt", "empirical", "theoretical", "relative_error"]].to_string(index=False))


def main() -> None:
    ensure_output_dirs()
    configure_matplotlib()

    linear_payload, linear_rows = run_linear_undamped_experiment(LinearUndampedConfig())
    damped_payload, damped_rows = run_linear_damped_experiment(LinearDampedConfig())
    quadratic_payload, quadratic_rows = run_quadratic_experiment(QuadraticConfig())
    recovery_payload, recovery_table = run_parameter_recovery()
    convergence_rows = run_convergence_checks()

    create_figure1(linear_payload)
    create_figure2(damped_payload)
    create_figure3(quadratic_payload)
    create_figure4(recovery_payload)
    create_figure5(linear_payload)

    verification_table = pd.DataFrame(linear_rows + damped_rows + quadratic_rows + convergence_rows)
    verification_path = RESULTS_DIR / "numerical_verification_summary.csv"
    verification_table.to_csv(verification_path, index=False)
    verification_table.to_csv(TABLES_DIR / "numerical_verification_summary.csv", index=False)
    recovery_path = RESULTS_DIR / "parameter_recovery_summary.csv"
    recovery_table.to_csv(recovery_path, index=False)
    recovery_table.to_csv(TABLES_DIR / "parameter_recovery_summary.csv", index=False)

    print_console_summary(verification_table)
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved summary tables to {RESULTS_DIR} and {TABLES_DIR}")


if __name__ == "__main__":
    main()
