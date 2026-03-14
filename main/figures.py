from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .utils import FIGURES_DIR


PALETTE = {
    "blue": "#1f4e79",
    "red": "#8c2d04",
    "green": "#3b6b4b",
    "gold": "#b58900",
    "gray": "#4d4d4d",
    "light": "#d9d9d9",
}


def _add_note(ax: plt.Axes, text: str, x: float = 0.03, y: float = 0.97) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8", "linewidth": 0.6},
    )


def _save_figure(fig: plt.Figure, filename: str) -> None:
    png_path = FIGURES_DIR / f"{filename}.png"
    pdf_path = FIGURES_DIR / f"{filename}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def create_figure1(payload: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(payload["time_sample"], payload["ua_sample"], color=PALETTE["blue"], label=r"$u_a$")
    ax.plot(payload["time_sample"], payload["uo_sample"], color=PALETTE["green"], label=r"$u_o$")
    ax.set_title("(a) Linear two-layer trajectories")
    ax.set_xlabel("time")
    ax.set_ylabel("velocity (dimensionless)")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.plot(payload["time_sample"], payload["us_sample"], color=PALETTE["red"])
    ax.axhline(0.0, color=PALETTE["gray"], lw=0.8)
    ax.set_title("(b) Mean reversion of the shear mode")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$u_s$")

    ax = axes[1, 0]
    ax.plot(payload["time_ensemble"], payload["ut_variance"], color=PALETTE["green"], label="simulation")
    ax.plot(
        payload["time_ensemble"],
        payload["ut_variance_theory"],
        color=PALETTE["gray"],
        ls="--",
        lw=1.0,
        label="theory",
    )
    ax.set_title("(c) Diffusive growth of the total mode")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\mathrm{Var}(u_T)$")
    ax.legend(frameon=False, loc="lower right")
    _add_note(ax, f"relative error = {100.0 * payload['total_growth_relerr']:.2f}%", x=0.04, y=0.94)

    ax = axes[1, 1]
    ax.plot(payload["time_stationary"], payload["running_var"], color=PALETTE["blue"], label="running estimate")
    ax.axhline(
        payload["shear_variance_theory"],
        color=PALETTE["gray"],
        ls="--",
        lw=1.0,
        label=r"theory: $R/(SM)$",
    )
    ax.set_title("(d) Verification of the exact shear variance law")
    ax.set_xlabel("time after burn-in")
    ax.set_ylabel(r"$\langle u_s^2\rangle$")
    ax.legend(frameon=False, loc="upper right")
    _add_note(ax, f"relative error = {100.0 * payload['shear_variance_relerr']:.2f}%", x=0.04, y=0.20)

    fig.subplots_adjust(wspace=0.28, hspace=0.32)
    _save_figure(fig, "figure_1_linear_undamped")


def create_figure2(payload: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(payload["time"], payload["ua"], color=PALETTE["blue"], label=r"$u_a$")
    ax.plot(payload["time"], payload["uo"], color=PALETTE["green"], label=r"$u_o$")
    ax.set_title("(a) Stationary trajectories with linear damping")
    ax.set_xlabel("time")
    ax.set_ylabel("velocity (dimensionless)")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    series_colors = [PALETTE["blue"], PALETTE["red"], PALETTE["green"]]
    for (key, values, theory, theory_label), color in zip(payload["covariance_series"], series_colors):
        ax.plot(payload["cov_times"], values, color=color, label=key)
        ax.axhline(theory, color=color, ls="--", lw=0.9, label=theory_label)
    ax.set_title("(b) Convergence to the Lyapunov covariance")
    ax.set_xlabel("time after burn-in")
    ax.set_ylabel("covariance")
    ax.legend(frameon=False, loc="center right", ncol=2)
    _add_note(ax, f"mean relative error = {100.0 * payload['covariance_mean_relerr']:.2f}%")

    ax = axes[1, 0]
    x = np.arange(3)
    width = 0.35
    ax.bar(
        x - width / 2.0,
        payload["empirical_entries"],
        width=width,
        label="empirical",
        color=PALETTE["blue"],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2.0,
        payload["theoretical_entries"],
        width=width,
        label="Lyapunov",
        color=PALETTE["light"],
        edgecolor=PALETTE["gray"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\Sigma_{aa}$", r"$\Sigma_{ao}$", r"$\Sigma_{oo}$"])
    ax.set_title("(c) Entrywise covariance agreement")
    ax.set_ylabel("value")
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 1]
    ax.scatter(
        payload["scatter_state"][:, 0],
        payload["scatter_state"][:, 1],
        s=4,
        alpha=0.08,
        color=PALETTE["blue"],
    )
    center = np.mean(payload["scatter_state"], axis=0)
    vals, vecs = np.linalg.eigh(payload["cov_theory"])
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    width, height = 2.0 * np.sqrt(np.maximum(vals, 1e-12)) * 2.0
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        lw=1.1,
        color=PALETTE["red"],
    )
    ax.add_patch(ellipse)
    ellipse.set_label("Lyapunov covariance ellipse")
    ax.set_title("(d) Stationary Gaussian structure")
    ax.set_xlabel(r"$u_a$")
    ax.set_ylabel(r"$u_o$")
    ax.legend(frameon=False, loc="upper right")

    fig.subplots_adjust(wspace=0.30, hspace=0.32)
    _save_figure(fig, "figure_2_linear_damped")


def create_figure3(payload: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(payload["time"], payload["ua"], color=PALETTE["blue"], label=r"$u_a$")
    ax.plot(payload["time"], payload["uo"], color=PALETTE["green"], label=r"$u_o$")
    ax.set_title("(a) Nonlinear trajectories under quadratic drag")
    ax.set_xlabel("time")
    ax.set_ylabel("velocity (dimensionless)")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.hist(
        payload["shear_hist"],
        bins=50,
        density=True,
        color=PALETTE["light"],
        edgecolor=PALETTE["gray"],
        label="simulation",
    )
    grid = np.linspace(np.min(payload["shear_hist"]), np.max(payload["shear_hist"]), 500)
    gaussian = (
        1.0
        / (payload["hist_std"] * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((grid - payload["hist_mean"]) / payload["hist_std"]) ** 2)
    )
    ax.plot(grid, gaussian, color=PALETTE["red"], ls="--", label="Gaussian fit")
    ax.axvline(0.0, color=PALETTE["gray"], lw=0.8)
    _add_note(
        ax,
        "\n".join(
            [
                rf"$\langle u_s^2\rangle={payload['shear_variance']:.3f}$",
                rf"$\langle |u_s|^3\rangle={payload['shear_abs3']:.3f}$",
            ]
        ),
    )
    ax.set_title("(b) Stationary shear distribution and Gaussian fit")
    ax.set_xlabel(r"$u_s$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 0]
    ax.plot(
        payload["sweep_index"],
        payload["normalized_lhs_values"],
        marker="o",
        color=PALETTE["blue"],
        label=r"$\tilde S M\langle |u_s|^3\rangle / R$",
    )
    ax.plot(
        payload["sweep_index"],
        payload["normalized_rhs_values"],
        marker="s",
        color=PALETTE["red"],
        label=payload["rhs_label"],
    )
    ax.set_xticks(payload["sweep_index"])
    ax.set_xticklabels(payload["sweep_labels"], rotation=25, ha="right")
    ax.set_title("(c) Normalized cubic-moment verification")
    ax.set_xlabel("parameter choice")
    ax.set_ylabel("normalized cubic moment")
    ax.legend(frameon=False, loc="best")
    _add_note(ax, f"relative error = {100.0 * payload['balance_relerr']:.2f}%")

    ax = axes[1, 1]
    ax.plot(payload["acf_lags"], payload["acf_nonlinear"], color=PALETTE["blue"], label="nonlinear")
    ax.plot(
        payload["acf_lags"],
        payload["acf_effective"],
        color=PALETTE["red"],
        ls="--",
        label=rf"effective OU ($S_{{eddy}}={payload['S_eddy']:.3f}$)",
    )
    ax.set_title("(d) Effective eddy-friction approximation")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.legend(frameon=False, loc="upper right")

    fig.subplots_adjust(wspace=0.30, hspace=0.34)
    _save_figure(fig, "figure_3_quadratic")


def create_figure5(payload: dict) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8))
    shear = payload["stationary_shear_hist"]
    ax.hist(
        shear,
        bins=60,
        density=True,
        color=PALETTE["light"],
        edgecolor=PALETTE["gray"],
        label="stationary histogram",
    )
    grid = np.linspace(np.min(shear), np.max(shear), 600)
    sigma = payload["boltzmann_std"]
    gaussian = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (grid / sigma) ** 2)
    ax.plot(
        grid,
        gaussian,
        color=PALETTE["red"],
        ls="--",
        label=r"analytic: $Z^{-1}\exp\!\left(-\frac{SM}{2R}u_s^2\right)$",
    )
    ax.set_title("Stationary distribution versus Boltzmann prediction")
    ax.set_xlabel(r"$u_s$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper right")
    _add_note(ax, f"relative error = {100.0 * payload['shear_variance_relerr']:.2f}%")
    fig.subplots_adjust(wspace=0.20, hspace=0.20)
    _save_figure(fig, "figure_5_boltzmann_fdr")


def create_figure4(payload: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.scatter(payload["true_S"], payload["recovered_S"], s=22, alpha=0.8, color=PALETTE["blue"])
    limits = [min(payload["true_S"]), max(payload["true_S"])]
    ax.plot(limits, limits, color=PALETTE["gray"], ls="--", lw=1.0, label="identity")
    fit = payload["fit_S"]
    xgrid = np.linspace(limits[0], limits[1], 100)
    ax.plot(xgrid, fit["slope"] * xgrid + fit["intercept"], color=PALETTE["red"], label="fit")
    ax.set_title("(a) Recovery of the coupling parameter $S$")
    ax.set_xlabel("true $S$")
    ax.set_ylabel("recovered $S$")
    ax.legend(frameon=False, loc="upper left")
    _add_note(
        ax,
        rf"slope = {fit['slope']:.3f}" + "\n" + rf"$R^2 = {fit['r_squared']:.4f}$",
        x=0.60,
        y=0.22,
    )

    ax = axes[0, 1]
    ax.scatter(payload["true_M"], payload["recovered_M"], s=22, alpha=0.8, color=PALETTE["green"])
    limits = [min(payload["true_M"]), max(payload["true_M"])]
    ax.plot(limits, limits, color=PALETTE["gray"], ls="--", lw=1.0, label="identity")
    fit = payload["fit_M"]
    xgrid = np.linspace(limits[0], limits[1], 100)
    ax.plot(xgrid, fit["slope"] * xgrid + fit["intercept"], color=PALETTE["red"], label="fit")
    ax.set_title("(b) Recovery of the inertial parameter $M$")
    ax.set_xlabel("true $M$")
    ax.set_ylabel("recovered $M$")
    ax.legend(frameon=False, loc="upper left")
    _add_note(
        ax,
        rf"slope = {fit['slope']:.3f}" + "\n" + rf"$R^2 = {fit['r_squared']:.4f}$",
        x=0.60,
        y=0.22,
    )

    ax = axes[1, 0]
    ax.boxplot(
        [payload["relerr_S"], payload["relerr_M"], payload["relerr_S_eddy"]],
        labels=[r"$S$", r"$M$", r"$S_{eddy}$"],
        showfliers=False,
    )
    ax.set_title("(c) Relative-error distributions")
    ax.set_ylabel("relative error")

    ax = axes[1, 1]
    im = ax.imshow(
        payload["heatmap"],
        origin="lower",
        aspect="auto",
        extent=payload["heatmap_extent"],
        cmap="viridis",
    )
    ax.set_title("(d) Mean recovery error over parameter space")
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$m$")
    fig.colorbar(im, ax=ax, shrink=0.9, label="mean relative error")

    fig.subplots_adjust(wspace=0.30, hspace=0.32)
    _save_figure(fig, "figure_4_parameter_recovery")
