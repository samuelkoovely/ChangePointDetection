import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = (
    BASE_DIR
    / 'gridsearch_results/multibkps_block2activities_snapshots_laplacians/gridsearch_results.pkl'
)
DATASET_PATH = BASE_DIR / 'data/multibkps_block2activities_snapshots.pkl'


def resolve_existing_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    candidates = [path, BASE_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return BASE_DIR / path


def get_best_signal_metadata(results: dict) -> tuple[int, int, list[list[float]], list[str]]:
    best_n_eigen = results.get('best_n_eigen')
    best_window_length = results.get('best_window_length')

    if best_n_eigen is None or best_window_length is None:
        best_index = results.get('best_index')
        if best_index is None:
            raise ValueError('No best n_eigen/window_length found in grid-search results.')
        best_n_eigen = int(results['n_eigens'][best_index[0]])
        best_window_length = int(results['window_lengths'][best_index[1]])

    best_result = None
    for n_eigen_result in results['n_eigen_results']:
        if int(n_eigen_result['n_eigen']) == int(best_n_eigen):
            best_result = n_eigen_result
            break

    if best_result is None:
        raise ValueError(f'Could not find n_eigen result for n_eigen={best_n_eigen}.')

    predicted_change_points = best_result['predicted_change_points'][int(best_window_length)]
    sample_names = best_result.get('sample_names')
    if sample_names is None:
        sample_names = [f'sample_{i}' for i in range(len(predicted_change_points))]

    return int(best_n_eigen), int(best_window_length), predicted_change_points, sample_names


def get_signals_outdir(results: dict) -> Path:
    signals_outdir = results.get('signals_outdir')
    if signals_outdir is None:
        return RESULTS_PATH.parent / 'signals'
    return resolve_existing_path(signals_outdir)


def get_signal_path(
    signals_outdir: Path,
    sample_name: str,
    n_eigen: int,
    window_length: int,
) -> Path:
    candidates = [
        signals_outdir
        / sample_name
        / f'signal_n_eigen_{int(n_eigen)}_window_length_{int(window_length)}.pkl',
        signals_outdir
        / sample_name
        / f'signal_n_eigen_{float(n_eigen):.11f}_window_{float(window_length):g}.pkl',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


with open(RESULTS_PATH, 'rb') as handle:
    results = pickle.load(handle)

with open(DATASET_PATH, 'rb') as handle:
    dataset = pickle.load(handle)

best_n_eigen, best_window_length, predicted_change_points, sample_names = get_best_signal_metadata(results)
signals_outdir = get_signals_outdir(results)
penalty = results.get('penalty')

n_samples = min(5, len(dataset), len(predicted_change_points))
fig, axes = plt.subplots(n_samples, 1, figsize=(14, 8), sharex=False)
axes = np.atleast_1d(axes)

for sample in range(n_samples):
    entry = dataset[sample]
    sample_name = sample_names[sample] if sample_names[sample] is not None else f'sample_{sample}'

    signal_path = get_signal_path(
        signals_outdir=signals_outdir,
        sample_name=sample_name,
        n_eigen=best_n_eigen,
        window_length=best_window_length,
    )
    with open(signal_path, 'rb') as handle:
        signal = pickle.load(handle)

    signal_values = np.asarray(signal['signal'])
    x_values = np.asarray(signal.get('snapshot_indices', np.arange(len(signal_values))))

    ax = axes[sample]
    ax.plot(x_values, signal_values, label=sample_name)
    if signal_values.size > 0:
        ymin = np.min(signal_values)
        ymax = np.max(signal_values)
        if np.isclose(ymin, ymax):
            ymax = ymin + 1.0

        for bkp in entry['bkps']:
            ax.vlines(
                float(bkp),
                ymin=ymin,
                ymax=ymax,
                color='black',
            )
        for pred_cp in predicted_change_points[sample]:
            ax.vlines(
                pred_cp,
                ymin=ymin,
                ymax=ymax,
                linestyles='dashed',
                color='red',
            )
    ax.legend(loc='upper right')
    ax.set_ylabel(f's{sample}')

title = f'Multi-bkps laplacians\nn_eigen={best_n_eigen}, window_length={best_window_length}'
if penalty is not None:
    title += f', penalty={float(penalty):g}'
axes[0].set_title(title)
axes[-1].set_xlabel('snapshot index')
fig.tight_layout()
plt.show()
