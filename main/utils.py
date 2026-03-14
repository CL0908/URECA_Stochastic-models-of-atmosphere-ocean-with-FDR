from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linewidth": 0.4,
            "grid.color": "#9aa0a6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.default": "it",
            "savefig.bbox": "tight",
        }
    )


def relative_error(empirical: float, theoretical: float) -> float:
    denom = max(abs(theoretical), 1e-12)
    return abs(empirical - theoretical) / denom


def centered_running_mean(values: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(values, dtype=float)
    counts = np.arange(1, values.size + 1, dtype=float)
    return cumulative / counts


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
