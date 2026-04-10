import json
import os
import pickle
from pathlib import Path

import numpy as np

from EDLDE import generate_smooth_SBM, trim_temporal_network_head_tail


REPO_ROOT = Path(__file__).resolve().parent
TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
MPL_CACHE_DIR = TMP_ROOT / "changepointdetection_mplconfig"
XDG_CACHE_DIR = TMP_ROOT / "changepointdetection_xdg_cache"

DEFAULT_RATES = (2, 5, 10)
DEFAULT_LAMBDAS = (0.1, 0.464158883, 2.15443469, 10.0)
DEFAULT_NETWORK_DIR = Path("data/edlde_upper_bound")
DEFAULT_RESULTS_DIR = Path("gridsearch_results/edlde_upper_bound")
DEFAULT_FIGURE_PATH = Path("figures/fig_global_entropy_edlde_upper_bound.pdf")
DEFAULT_TIME_INTERVALS = ((0, 100),)
DEFAULT_SEED_HINTS = {
    2: 4,
    5: 1,
    10: 1,
}
RATE_CONFIGS = {
    2: {
        "density": 5.0,
        "inter_tau": 5.0,
        "list_p_within_community": [1.0],
    },
    5: {
        "density": 5.0,
        "inter_tau": 5.0,
    },
    10: {
        "density": 10.0,
        "inter_tau": 10.0,
    },
}
N_GROUPS = 4
N_PER_GROUP = 25
TOTAL_NODES = N_GROUPS * N_PER_GROUP
T_START = 0.0
T_END = 100.0
INTER_TAU = 5.0
POWER_CONFIG = [1]
BASIS_NUM_COMMUNITIES = 4
PANEL_TITLES = {
    2: "(A) 5 links/s, no noise",
    5: "(B) 5 links/s",
    10: "(C) 10 links/s",
}
INSET_POSITIONS = (0.37,)


def ensure_plotting_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def rate_slug(rate: int | float) -> str:
    return f"rate_{float(rate):g}".replace(".", "p")


def preferred_seed_order(rate: int) -> list[int]:
    seeds = []
    hint = DEFAULT_SEED_HINTS.get(int(rate))
    if hint is not None:
        seeds.append(int(hint))
    seeds.extend(seed for seed in range(1, 101) if seed != hint)
    return seeds


def rate_config(rate: int) -> dict:
    config = dict(RATE_CONFIGS.get(int(rate), {}))
    config.setdefault("density", float(rate))
    config.setdefault("inter_tau", INTER_TAU)
    return config


def panel_title(rate: int) -> str:
    return PANEL_TITLES.get(int(rate), f"{float(rate):g} links/s")


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_pickle(path_like: str | Path):
    path = resolve_repo_path(path_like)
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(obj, path_like: str | Path) -> None:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def sample_rate_network(rate: int, seed: int):
    config = rate_config(rate)
    net = generate_smooth_SBM(
        density=float(config["density"]),
        inter_tau=float(config["inter_tau"]),
        n_per_group=N_PER_GROUP,
        n_groups=N_GROUPS,
        t_start=T_START,
        t_end=T_END,
        basis_num_communities=BASIS_NUM_COMMUNITIES,
        powers_num_communities=POWER_CONFIG,
        list_p_within_community=config.get("list_p_within_community"),
        seed=int(seed),
    )
    trimmed_net, _ = trim_temporal_network_head_tail(
        temporal_net=net,
        density=float(config["density"]),
        inter_tau=float(config["inter_tau"]),
        tail_start_time=T_END,
        head_start_time=T_START,
        clip_ending_times=True,
        align_first_event_to_zero=False,
    )
    return trimmed_net


def ensure_sampled_network(rate: int, network_dir: Path, force: bool):
    rate_dir = network_dir / rate_slug(rate)
    network_path = rate_dir / "network.pkl"
    metadata_path = rate_dir / "metadata.json"

    if network_path.exists() and metadata_path.exists() and not force:
        return load_rate_network(rate=rate, network_dir=network_dir)

    rate_dir.mkdir(parents=True, exist_ok=True)
    selected_seed = None
    net = None
    config = rate_config(rate)

    for seed in preferred_seed_order(rate):
        candidate = sample_rate_network(rate=rate, seed=seed)
        if candidate.num_nodes == TOTAL_NODES and np.isclose(float(candidate.times[-1]), T_END):
            selected_seed = seed
            net = candidate
            break

    if net is None:
        raise RuntimeError(
            f"Could not sample a {TOTAL_NODES}-node network for rate={rate} "
            "within the tested seed range."
        )

    metadata = {
        "rate": int(rate),
        "density": float(config["density"]),
        "seed": int(selected_seed),
        "t_start": T_START,
        "t_end": T_END,
        "inter_tau": float(config["inter_tau"]),
        "list_p_within_community": config.get("list_p_within_community"),
        "n_groups": N_GROUPS,
        "n_per_group": N_PER_GROUP,
        "basis_num_communities": BASIS_NUM_COMMUNITIES,
        "powers_num_communities": POWER_CONFIG,
        "num_nodes": int(net.num_nodes),
        "num_events": int(net.num_events),
        "num_times": int(len(net.times)),
        "network_path": str(network_path),
    }

    dump_pickle(net, network_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return net, metadata


def load_rate_network_metadata(rate: int, network_dir: Path) -> dict:
    metadata_path = resolve_repo_path(network_dir / rate_slug(rate) / "metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing network metadata for rate {rate} at {metadata_path}. "
            "Run generate_edlde_upper_bound_networks.py first."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_rate_network(rate: int, network_dir: Path):
    metadata = load_rate_network_metadata(rate=rate, network_dir=network_dir)
    net = load_pickle(metadata["network_path"])
    return net, metadata


def load_networks_and_metadata(rates, network_dir: Path):
    networks = {}
    metadata_by_rate = {}
    for rate in rates:
        net, metadata = load_rate_network(rate=rate, network_dir=network_dir)
        networks[int(rate)] = net
        metadata_by_rate[str(int(rate))] = metadata
    return networks, metadata_by_rate


def signal_filename(lamda: float) -> str:
    return f"global_S{float(lamda):.11f}"


def rate_results_dir(rate: int, results_dir: Path) -> Path:
    return results_dir / rate_slug(rate)


def rate_signal_dir(rate: int, results_dir: Path) -> Path:
    return rate_results_dir(rate=rate, results_dir=results_dir) / "global_S"


def rate_limit_path(rate: int, results_dir: Path) -> Path:
    return rate_results_dir(rate=rate, results_dir=results_dir) / "global_limit_selected" / "global_limit.pkl"


def summary_path(results_dir: Path) -> Path:
    return resolve_repo_path(results_dir / "summary.json")


def write_summary(
    rates,
    lambdas,
    network_dir: Path,
    results_dir: Path,
    figure_path: Path | None = None,
    metadata_by_rate: dict | None = None,
) -> Path:
    summary_filepath = summary_path(results_dir)
    existing = {}
    if summary_filepath.exists():
        existing = json.loads(summary_filepath.read_text(encoding="utf-8"))

    if metadata_by_rate is None:
        metadata_by_rate = {
            str(int(rate)): load_rate_network_metadata(rate=rate, network_dir=network_dir)
            for rate in rates
        }

    chosen_figure_path = (
        str(figure_path)
        if figure_path is not None
        else existing.get("figure_path", str(DEFAULT_FIGURE_PATH))
    )
    summary = {
        "rates": [int(rate) for rate in rates],
        "lambdas": [float(lamda) for lamda in lambdas],
        "rate_configs": {str(int(rate)): rate_config(rate) for rate in rates},
        "network_dir": str(network_dir),
        "results_dir": str(results_dir),
        "figure_path": chosen_figure_path,
        "networks": metadata_by_rate,
    }
    summary_filepath.parent.mkdir(parents=True, exist_ok=True)
    summary_filepath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_filepath
