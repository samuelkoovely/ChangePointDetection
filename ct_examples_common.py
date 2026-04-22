from __future__ import annotations

import pickle
import random as rd
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import EDLDE
from signal_generation import compute_signals_for_lambda


DEFAULT_LAMDA = 1.0
DEFAULT_ENTROPY_WINDOWS = [0.25, 2.0, 10.0]
DEFAULT_DATA_DIR = Path("data/ct_examples")
DEFAULT_FIGURE_DIR = Path("figures")
DEFAULT_CT_NUM_SAMPLES = 10
DEFAULT_CT_RANDOM_SEED = 34
DEFAULT_CT_TEST_NUM_SAMPLES = 50
DEFAULT_CT_TEST_RANDOM_SEED = 1034
DEFAULT_CT_TEST_SAMPLE_INDEX_OFFSET = 1000

DEFAULT_BREAKPOINT_FRACTION_LOWER = 0.25
DEFAULT_BREAKPOINT_FRACTION_UPPER = 0.75

BLOCK1_EVENT_SEED_BASE = 1415
BLOCK2_PHASE_1_EVENT_SEED_BASE = 1415
BLOCK2_PHASE_2_EVENT_SEED_BASE = 8281
SBM_SEED_BASE = 271


@dataclass(frozen=True)
class SparseSampleSpec:
    key: str
    title: str
    sample_filename: str
    signal_filename: str
    figure_filename: str


SPECS = (
    SparseSampleSpec(
        key="block1",
        title="Sparse Block-1 activity",
        sample_filename="block1activity_sparse_sample.pkl",
        signal_filename="block1activity_sparse_signal.pkl",
        figure_filename="block1activity_sparse_local_entropy_panels.png",
    ),
    SparseSampleSpec(
        key="block2",
        title="Sparse Block-2 activities",
        sample_filename="block2activities_sparse_sample.pkl",
        signal_filename="block2activities_sparse_signal.pkl",
        figure_filename="block2activities_sparse_local_entropy_panels.png",
    ),
)

SPEC_BY_KEY = {spec.key: spec for spec in SPECS}


def get_specs(keys: list[str] | tuple[str, ...]) -> list[SparseSampleSpec]:
    return [SPEC_BY_KEY[key] for key in keys]


def sample_path(spec: SparseSampleSpec, data_dir: Path) -> Path:
    return data_dir / spec.sample_filename


def signal_path(spec: SparseSampleSpec, data_dir: Path) -> Path:
    return data_dir / spec.signal_filename


def figure_path(spec: SparseSampleSpec, figure_dir: Path) -> Path:
    return figure_dir / spec.figure_filename


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_windows(windows: list[float] | tuple[float, ...]) -> list[float]:
    ordered_unique = []
    seen = set()
    for window in windows:
        value = float(window)
        if value <= 0:
            raise ValueError("All entropy windows must be strictly positive.")
        if value not in seen:
            seen.add(value)
            ordered_unique.append(value)
    return ordered_unique


def _default_breakpoint_bounds(t_start: float, t_end: float) -> tuple[int, int]:
    interval = float(t_end) - float(t_start)
    lower = float(t_start) + DEFAULT_BREAKPOINT_FRACTION_LOWER * interval
    upper = float(t_start) + DEFAULT_BREAKPOINT_FRACTION_UPPER * interval
    return int(round(lower)), int(round(upper))


def _build_trimmed_sparse_sample(
    tnet: Any,
    t_split: float,
    *,
    trim_density: float,
    trim_inter_tau: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    tnet_trimmed, t_0 = EDLDE.trim_temporal_network_head_tail(
        tnet,
        density=trim_density,
        inter_tau=trim_inter_tau,
        tail_start_time=metadata["t_end"],
    )

    breakpoint = float(t_split - t_0)
    return {
        "tnet": tnet_trimmed,
        "bkps": [breakpoint],
        "n_bkps": 1,
        "starts": tnet_trimmed.events_table["starting_times"],
        "ends": tnet_trimmed.events_table["ending_times"],
        "metadata": metadata,
    }


def block1_sparse_ct_parameters() -> dict[str, Any]:
    return {
        "model": "block1_sparse",
        "density": 30,
        "inter_tau": 5,
        "t_start": 0,
        "t_end": 100,
        "n_per_group": 50,
        "n_groups": 4,
        "basis_num_communities": 2,
        "powers_num_communities": [1, 0],
        "list_p_within_community": [49 / 50, 49 / 50],
    }


def block2_sparse_ct_parameters() -> dict[str, Any]:
    return {
        "model": "block2_sparse",
        "density_1": 20,
        "density_2": 40,
        "inter_tau_1": 5,
        "inter_tau_2": 5,
        "t_start": 0,
        "t_end": 100,
        "n_per_group": 200,
        "n_groups": 1,
        "basis_num_communities": 2,
        "powers_num_communities": [0],
        "list_p_within_community": [49 / 50],
    }


def generate_block1_sparse_ct_sample(
    sample_index: int = 0,
    t_split: float | None = None,
) -> dict[str, Any]:
    params = block1_sparse_ct_parameters()
    if t_split is None:
        t_split = 0.5 * (params["t_start"] + params["t_end"])

    number_of_events, starting_times, ending_times = EDLDE.EDLDE(
        density=params["density"],
        inter_tau=params["inter_tau"],
        t_start=params["t_start"],
        t_end=params["t_end"],
        seed=BLOCK1_EVENT_SEED_BASE + sample_index,
    )

    tnet = EDLDE.generate_smooth_SBM(
        inter_tau=0,
        density=0,
        n_per_group=params["n_per_group"],
        n_groups=params["n_groups"],
        t_start=params["t_start"],
        t_end=params["t_end"],
        basis_num_communities=params["basis_num_communities"],
        powers_num_communities=params["powers_num_communities"],
        list_p_within_community=params["list_p_within_community"],
        breakpoints=[t_split],
        number_of_events=number_of_events,
        starting_times=starting_times,
        ending_times=ending_times,
        seed=SBM_SEED_BASE + sample_index,
    )

    metadata = dict(params)
    metadata["t_split"] = float(t_split)
    return _build_trimmed_sparse_sample(
        tnet,
        t_split=float(t_split),
        trim_density=params["density"],
        trim_inter_tau=params["inter_tau"],
        metadata=metadata,
    )


def generate_block1_sparse_sample() -> dict[str, Any]:
    return generate_block1_sparse_ct_sample(sample_index=0, t_split=50)


def generate_piecewise_sparse_activity(
    t_split: float,
    t_end: float,
    density_1: float,
    density_2: float,
    inter_tau_1: float,
    inter_tau_2: float,
    sample_index: int = 0,
) -> tuple[int, np.ndarray, np.ndarray]:
    n_phase_1, starts_phase_1, ends_phase_1 = EDLDE.EDLDE(
        density=density_1,
        inter_tau=inter_tau_1,
        t_start=0,
        t_end=t_split,
        seed=BLOCK2_PHASE_1_EVENT_SEED_BASE + sample_index,
    )
    n_phase_2, starts_phase_2, ends_phase_2 = EDLDE.EDLDE(
        density=density_2,
        inter_tau=inter_tau_2,
        t_start=t_split,
        t_end=t_end,
        seed=BLOCK2_PHASE_2_EVENT_SEED_BASE + sample_index,
    )

    starts = np.concatenate([starts_phase_1, starts_phase_2])
    ends = np.concatenate([ends_phase_1, ends_phase_2])
    return n_phase_1 + n_phase_2, starts, ends


def generate_block2_sparse_ct_sample(
    sample_index: int = 0,
    t_split: float | None = None,
) -> dict[str, Any]:
    params = block2_sparse_ct_parameters()
    if t_split is None:
        t_split = 0.5 * (params["t_start"] + params["t_end"])

    number_of_events, starting_times, ending_times = generate_piecewise_sparse_activity(
        t_split=float(t_split),
        t_end=params["t_end"],
        density_1=params["density_1"],
        density_2=params["density_2"],
        inter_tau_1=params["inter_tau_1"],
        inter_tau_2=params["inter_tau_2"],
        sample_index=sample_index,
    )

    tnet = EDLDE.generate_smooth_SBM(
        inter_tau=0,
        density=0,
        n_per_group=params["n_per_group"],
        n_groups=params["n_groups"],
        t_start=params["t_start"],
        t_end=params["t_end"],
        basis_num_communities=params["basis_num_communities"],
        powers_num_communities=params["powers_num_communities"],
        list_p_within_community=params["list_p_within_community"],
        number_of_events=number_of_events,
        starting_times=starting_times,
        ending_times=ending_times,
        seed=SBM_SEED_BASE + sample_index,
    )

    metadata = dict(params)
    metadata["t_split"] = float(t_split)
    return _build_trimmed_sparse_sample(
        tnet,
        t_split=float(t_split),
        trim_density=params["density_1"],
        trim_inter_tau=params["inter_tau_1"],
        metadata=metadata,
    )


def generate_block2_sparse_sample() -> dict[str, Any]:
    return generate_block2_sparse_ct_sample(sample_index=0, t_split=50)


def generate_block1_sparse_ct_dataset(
    num_samples: int = DEFAULT_CT_NUM_SAMPLES,
    random_seed: int = DEFAULT_CT_RANDOM_SEED,
    sample_index_offset: int = 0,
) -> list[dict[str, Any]]:
    params = block1_sparse_ct_parameters()
    lower, upper = _default_breakpoint_bounds(params["t_start"], params["t_end"])
    rng = rd.Random(random_seed)
    return [
        generate_block1_sparse_ct_sample(
            sample_index=sample_index_offset + sample_number,
            t_split=rng.randint(lower, upper),
        )
        for sample_number in range(num_samples)
    ]


def generate_block2_sparse_ct_dataset(
    num_samples: int = DEFAULT_CT_NUM_SAMPLES,
    random_seed: int = DEFAULT_CT_RANDOM_SEED,
    sample_index_offset: int = 0,
) -> list[dict[str, Any]]:
    params = block2_sparse_ct_parameters()
    lower, upper = _default_breakpoint_bounds(params["t_start"], params["t_end"])
    rng = rd.Random(random_seed)
    return [
        generate_block2_sparse_ct_sample(
            sample_index=sample_index_offset + sample_number,
            t_split=rng.randint(lower, upper),
        )
        for sample_number in range(num_samples)
    ]


def generate_sparse_sample(spec_key: str) -> dict[str, Any]:
    if spec_key == "block1":
        return generate_block1_sparse_sample()
    if spec_key == "block2":
        return generate_block2_sparse_sample()
    raise ValueError(f"Unsupported sparse sample key: {spec_key!r}")


def compute_entropy_signals(
    sample: dict[str, Any],
    lamda: float,
    windows: list[float],
) -> dict[float, dict[str, Any]]:
    net = sample["tnet"]
    results = compute_signals_for_lambda(
        net=net,
        lamda=float(lamda),
        windows=windows,
        sample_fraction=1.0,
        p0=None,
        use_linear_approx=False,
        lin_t_s=10,
        window_backend="segment_tree",
        reverse_time=False,
    )
    return {float(window): results[float(window)] for window in windows}


def build_signal_bundle(
    signals_by_window: dict[float, dict[str, Any]],
    lamda: float,
    windows: list[float],
) -> dict[str, Any]:
    return {
        "lamda": float(lamda),
        "windows": np.asarray(windows, dtype=float),
        "signals_by_window": signals_by_window,
    }
