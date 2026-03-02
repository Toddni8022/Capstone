"""Plotting and visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capstone.config import FIGURES_DIR


def plot_housing_map(
    data: pd.DataFrame,
    save: bool = False,
    filename: str = "housing_map.png",
    figures_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    """Plot a geographical scatter of housing prices.

    Parameters
    ----------
    data:
        Housing DataFrame with ``longitude``, ``latitude``,
        ``population``, and ``median_house_value`` columns.
    save:
        Whether to save the figure to disk.
    filename:
        Output filename when ``save=True``.
    figures_dir:
        Directory where the figure is saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        data["longitude"],
        data["latitude"],
        alpha=0.4,
        s=data["population"] / 100,
        c=data["median_house_value"],
        cmap="jet",
        label="Population",
    )
    plt.colorbar(scatter, label="Median House Value ($)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("California Housing Prices")
    ax.legend()

    if save:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    save: bool = False,
    filename: str = "correlation_matrix.png",
    figures_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    """Plot a heatmap-style correlation matrix for numerical columns.

    Parameters
    ----------
    data:
        DataFrame containing numerical features.
    save:
        Whether to save the figure to disk.
    filename:
        Output filename when ``save=True``.
    figures_dir:
        Directory where the figure is saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    corr_matrix = data.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()

    if save:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importances(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    save: bool = False,
    filename: str = "feature_importances.png",
    figures_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    """Plot a horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names:
        List of feature names.
    importances:
        Array of importance scores corresponding to ``feature_names``.
    top_n:
        Number of top features to display.
    save:
        Whether to save the figure to disk.
    filename:
        Output filename when ``save=True``.
    figures_dir:
        Directory where the figure is saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_importances[::-1], align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()

    if save:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = False,
    filename: str = "predictions_vs_actual.png",
    figures_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    """Scatter plot of predicted values against actual values.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    save:
        Whether to save the figure to disk.
    filename:
        Output filename when ``save=True``.
    figures_dir:
        Directory where the figure is saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Values ($)")
    ax.set_ylabel("Predicted Values ($)")
    ax.set_title("Predictions vs Actual")
    ax.legend()
    fig.tight_layout()

    if save:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")

    return fig
