from __future__ import annotations

from pathlib import Path


PRIMARY_SCHOOL_WINDOW_SECONDS = 30.0 * 60.0
PRIMARY_SCHOOL_WINDOW_MINUTES = PRIMARY_SCHOOL_WINDOW_SECONDS / 60.0

# The saved grid uses the nearest available value to the requested 1.67e-03 rate.
PRIMARY_SCHOOL_TARGET_LAMDA = 1.67e-03
PRIMARY_SCHOOL_LAMDA = 0.00166810054
PRIMARY_SCHOOL_LAMDA_KEY = f"{PRIMARY_SCHOOL_LAMDA:.11f}"

PRIMARY_SCHOOL_RUPTURES_PENALTIES = (
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    21.0,
    34.0,
    55.0,
    89.0,
)
# Default downstream interval selection for primary-school clustering uses the
# 30-minute linear-kernel ruptures run at penalty 8.
PRIMARY_SCHOOL_DEFAULT_PENALTY = 8.0
PRIMARY_SCHOOL_DERIVATIVE_RUPTURES_PENALTIES = (
    0.1,
    0.3,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    21.0,
    34.0,
    55.0,
)
PRIMARY_SCHOOL_DERIVATIVE_SAVGOL_WINDOW_LENGTH = 31
PRIMARY_SCHOOL_DERIVATIVE_SAVGOL_POLYORDER = 3


def build_primary_school_window_lambda_dir(
    base_dir: Path,
    relative_root: str,
    reverse_time: bool = False,
) -> Path:
    direction = "backward" if reverse_time else "forward"
    return (
        base_dir
        / "gridsearch_results"
        / relative_root
        / direction
        / f"window_{int(PRIMARY_SCHOOL_WINDOW_SECONDS)}"
        / f"lamda_{PRIMARY_SCHOOL_LAMDA_KEY}"
    )


def build_primary_school_signal_path(
    signal_base: Path,
    reverse_time: bool = False,
) -> Path:
    signal_subdir = "window_S_selected_rev" if reverse_time else "window_S_selected"
    return (
        signal_base
        / signal_subdir
        / str(int(PRIMARY_SCHOOL_WINDOW_SECONDS))
        / f"window_S{PRIMARY_SCHOOL_LAMDA_KEY}"
    )


def build_primary_school_ruptures_results_path(
    base_dir: Path,
    reverse_time: bool = False,
) -> Path:
    return (
        build_primary_school_window_lambda_dir(
            base_dir=base_dir,
            relative_root="primaryschool_day1_ruptures",
            reverse_time=reverse_time,
        )
        / "ruptures_results.pkl"
    )


def build_primary_school_derivative_signal_path(
    base_dir: Path,
    reverse_time: bool = False,
) -> Path:
    return (
        build_primary_school_window_lambda_dir(
            base_dir=base_dir,
            relative_root="primaryschool_day1_derivative",
            reverse_time=reverse_time,
        )
        / "entropy_derivative.pkl"
    )


def build_primary_school_derivative_plot_path(
    base_dir: Path,
    reverse_time: bool = False,
) -> Path:
    return (
        build_primary_school_window_lambda_dir(
            base_dir=base_dir,
            relative_root="primaryschool_day1_derivative",
            reverse_time=reverse_time,
        )
        / "entropy_derivative_signal.pdf"
    )


def build_primary_school_derivative_ruptures_results_path(
    base_dir: Path,
    reverse_time: bool = False,
) -> Path:
    return (
        build_primary_school_window_lambda_dir(
            base_dir=base_dir,
            relative_root="primaryschool_day1_derivative_ruptures",
            reverse_time=reverse_time,
        )
        / "ruptures_results.pkl"
    )
