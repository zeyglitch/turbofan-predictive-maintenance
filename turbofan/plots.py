"""Figures attached to the MLflow runs."""

import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, title: str, cap: int):
    """Scatter plot of predicted against true remaining life; a perfect model sits on the line."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=3, alpha=0.3)
    ax.plot([0, cap], [0, cap], color="black", linewidth=1)
    ax.set_xlabel("true rul_clipped")
    ax.set_ylabel(f"predicted (bounded to [0, {cap}])")
    ax.set_title(title)
    return fig


def plot_importance(importance, title: str, xlabel: str):
    """Horizontal bars, the most important input at the top."""
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(importance) + 1))
    ax.barh(importance.index[::-1], importance.to_numpy()[::-1])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig
