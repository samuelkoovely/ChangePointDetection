from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import auxiliary_functions
import fig_gridsearch_primaryschool_limit as primaryschool_limit_fig
import primary_school_limit

plt.style.use(Path(__file__).with_name("paper.mplstyle"))


OUTPUT_FIGURE = (
    primaryschool_limit_fig.FIGURES_DIR
    / "fig_gridsearch_primaryschool_fastest_bounds.pdf"
)
SIGNAL_LINEWIDTH = 1.9
BOUND_LINEWIDTH = 1.8
UPPER_BOUND_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": BOUND_LINEWIDTH,
    "zorder": 5,
}
LOWER_BOUND_STYLE = {
    "color": "0.45",
    "linestyle": "-.",
    "linewidth": BOUND_LINEWIDTH,
    "zorder": 5,
}


def get_lambdas(signal_metadata: dict | None) -> np.ndarray:
    """
    Return all available primary-school diffusion rates.
    """

    if signal_metadata is None or "lambdas" not in signal_metadata:
        return np.asarray(primaryschool_limit_fig.DEFAULT_LAMBDAS, dtype=float)

    return np.asarray(signal_metadata["lambdas"], dtype=float)


def get_fastest_lambda(signal_metadata: dict | None) -> tuple[float, str]:
    """
    Return the largest saved lambda and the color matching the full lambda grid.
    """

    lambdas = get_lambdas(signal_metadata)
    if len(lambdas) == 0:
        raise ValueError("No lambdas are available for the primary-school signal.")

    fastest_idx = int(np.argmax(lambdas))
    colors = auxiliary_functions.generate_plasma_colors(len(lambdas))
    return float(lambdas[fastest_idx]), colors[fastest_idx]


def extract_lower_bound_array(payload: dict) -> np.ndarray:
    """
    Extract the connected-component lower-bound curve.
    """

    if "time_component_lower_bound_sums" in payload:
        return np.asarray(payload["time_component_lower_bound_sums"], dtype=float)

    return np.column_stack(
        (
            np.asarray(payload["t_samples"], dtype=float),
            np.asarray(payload["signal_array"], dtype=float),
        )
    )


def compute_lower_bound_payloads(windows_seconds: list[float]) -> dict[float, dict]:
    """
    Compute lower-bound curves for the requested primary-school windows.
    """

    day1_net, _ = primary_school_limit.load_primary_school_day1_network()
    window_plans = primary_school_limit.prepare_full_window_scan(
        day1_net,
        windows_seconds=windows_seconds,
    )

    return {
        float(window_seconds): primary_school_limit.compute_window_lower_bound_curve(
            net=day1_net,
            plan=window_plans[float(window_seconds)],
        )
        for window_seconds in windows_seconds
    }


def main() -> None:
    signal_metadata = primaryschool_limit_fig.load_metadata(
        primaryschool_limit_fig.SIGNAL_OUTPUT_BASE
    )
    windows_seconds = primaryschool_limit_fig.get_windows_seconds(signal_metadata)
    fastest_lambda, signal_color = get_fastest_lambda(signal_metadata)
    lower_bound_payloads = compute_lower_bound_payloads(windows_seconds)

    fig, axes = plt.subplots(
        1,
        len(windows_seconds),
        figsize=(15.5, 6.0),
        sharey=False,
    )
    if len(windows_seconds) == 1:
        axes = [axes]

    for ax, window_seconds in zip(axes, windows_seconds):
        signal_payload = primaryschool_limit_fig.load_signal_payload(
            window_seconds=window_seconds,
            lamda=fastest_lambda,
        )
        t_samples = np.asarray(signal_payload["t_samples"], dtype=float) / 3600.0
        signal = primaryschool_limit_fig.extract_signal_array(
            signal_payload,
            lamda=fastest_lambda,
        )
        ax.plot(
            t_samples,
            signal,
            color=signal_color,
            linewidth=SIGNAL_LINEWIDTH,
        )

        lower_curve = extract_lower_bound_array(
            lower_bound_payloads[float(window_seconds)]
        )
        ax.plot(
            lower_curve[:, 0] / 3600.0,
            lower_curve[:, 1],
            **LOWER_BOUND_STYLE,
        )

        upper_payload = primaryschool_limit_fig.load_limit_payload(
            window_seconds=window_seconds
        )
        upper_curve = primaryschool_limit_fig.extract_limit_array(upper_payload)
        ax.plot(
            upper_curve[:, 0] / 3600.0,
            upper_curve[:, 1],
            **UPPER_BOUND_STYLE,
        )

        ax.set_title(
            primaryschool_limit_fig.window_title(window_seconds),
            fontsize=primaryschool_limit_fig.PANEL_TITLE_FONTSIZE,
        )
        ax.set_xlabel(
            "Time (hours)",
            fontsize=primaryschool_limit_fig.AXIS_LABEL_FONTSIZE,
        )
        ax.tick_params(
            axis="both",
            labelsize=primaryschool_limit_fig.TICK_LABEL_FONTSIZE,
        )
        ax.set_box_aspect(1)

    axes[0].set_ylabel(
        "Entropy",
        fontsize=primaryschool_limit_fig.AXIS_LABEL_FONTSIZE,
    )

    signal_label = (
        f"Entropy, {primaryschool_limit_fig.format_lambda_label(fastest_lambda)}"
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=signal_color,
            linewidth=SIGNAL_LINEWIDTH,
            label=signal_label,
        ),
        Line2D([0], [0], label="Lower bound", **LOWER_BOUND_STYLE),
        Line2D([0], [0], label="Upper bound", **UPPER_BOUND_STYLE),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        fontsize=primaryschool_limit_fig.LEGEND_FONTSIZE,
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    primaryschool_limit_fig.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
