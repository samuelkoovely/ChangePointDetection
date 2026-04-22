import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / 'figures' / f'{Path(__file__).stem}.pdf'
RESULTS_PATH = (
    BASE_DIR
    / 'gridsearch_results/multibkps_block2activities_snapshots_frobenius/gridsearch_results.pkl'
)
DATASET_PATH = BASE_DIR / 'data/multibkps_block2activities_snapshots.pkl'


def resolve_existing_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    candidates = [path, BASE_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return BASE_DIR / path


def find_matching_float_key(mapping: dict, target: float, name: str) -> float:
    for key in mapping:
        if np.isclose(float(key), float(target)):
            return float(key)
    raise ValueError(f'Could not find {name}={target} in saved results.')


def find_matching_int_key(mapping: dict, target: int, name: str) -> int:
    for key in mapping:
        if int(key) == int(target):
            return int(key)
    raise ValueError(f'Could not find {name}={target} in saved results.')


def get_best_signal_metadata(
    results: dict,
) -> tuple[int, float | None, list[list[float]], list[str]]:
    best_window_length = results.get('best_window_length')
    best_penalty = results.get('best_penalty', results.get('penalty'))

    if best_window_length is None:
        best_index = results.get('best_index')
        if best_index is None:
            raise ValueError('No best window_length found in grid-search results.')
        best_window_length = int(results['window_lengths'][best_index[0]])
        penalties = results.get('penalties')
        if best_penalty is None and penalties is not None and len(best_index) > 1:
            best_penalty = float(penalties[best_index[1]])

    best_result = None
    results_by_penalty = results.get('results_by_penalty')
    if isinstance(results_by_penalty, dict) and best_penalty is not None:
        penalty_key = find_matching_float_key(results_by_penalty, float(best_penalty), 'penalty')
        best_result = results_by_penalty[penalty_key]

    if best_result is None:
        penalty_results = results.get('penalty_results', [])
        for penalty_result in penalty_results:
            result_penalty = penalty_result.get('penalty')
            penalty_matches = (
                best_penalty is None
                or (
                    result_penalty is not None
                    and np.isclose(float(result_penalty), float(best_penalty))
                )
            )
            if penalty_matches:
                best_result = penalty_result
                break

    if best_result is None:
        best_result = results.get('window_results')

    if best_result is None:
        raise ValueError(
            'Could not find the best window_length/penalty combination in grid-search results.'
        )

    predicted_change_points_by_window = best_result['predicted_change_points']
    window_key = find_matching_int_key(
        predicted_change_points_by_window,
        int(best_window_length),
        'window_length',
    )
    predicted_change_points = predicted_change_points_by_window[window_key]
    sample_names = best_result.get('sample_names')
    if sample_names is None:
        sample_names = [f'sample_{i}' for i in range(len(predicted_change_points))]

    return (
        int(best_window_length),
        None if best_penalty is None else float(best_penalty),
        predicted_change_points,
        sample_names,
    )


def get_signals_outdir(results: dict) -> Path:
    signals_outdir = results.get('signals_outdir')
    if signals_outdir is None:
        return RESULTS_PATH.parent / 'signals'
    return resolve_existing_path(signals_outdir)


def get_signal_path(signals_outdir: Path, sample_name: str, window_length: int) -> Path:
    candidates = [
        signals_outdir
        / sample_name
        / f'signal_window_length_{int(window_length)}.pkl',
        signals_outdir
        / sample_name
        / f'signal_window_{float(window_length):g}.pkl',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


with open(RESULTS_PATH, 'rb') as handle:
    results = pickle.load(handle)

with open(DATASET_PATH, 'rb') as handle:
    dataset = pickle.load(handle)

best_window_length, best_penalty, predicted_change_points, sample_names = get_best_signal_metadata(results)
signals_outdir = get_signals_outdir(results)

n_samples = min(len(dataset), len(predicted_change_points))
if n_samples <= 0:
    raise ValueError('No samples available to plot.')
fig, axes = plt.subplots(n_samples, 1, figsize=(14, max(2.2 * n_samples, 4)), sharex=False)
axes = np.atleast_1d(axes)
legend_handles = [
    Line2D([0], [0], color='C0', linewidth=1.5, label='Signal'),
    Line2D([0], [0], color='black', linewidth=1.5, label='Change point'),
    Line2D([0], [0], color='red', linewidth=1.5, linestyle='dashed', label='Predicted change point'),
]

for sample in range(n_samples):
    entry = dataset[sample]
    sample_name = sample_names[sample] if sample_names[sample] is not None else f'sample_{sample}'

    signal_path = get_signal_path(
        signals_outdir=signals_outdir,
        sample_name=sample_name,
        window_length=best_window_length,
    )
    with open(signal_path, 'rb') as handle:
        signal = pickle.load(handle)

    signal_values = np.asarray(signal['signal'])
    x_values = np.asarray(signal.get('snapshot_indices', np.arange(len(signal_values))))

    ax = axes[sample]
    ax.plot(x_values, signal_values, color='C0')
    if signal_values.size > 0:
        ymin = np.min(signal_values)
        ymax = np.max(signal_values)
        if np.isclose(ymin, ymax):
            pad = max(abs(float(ymin)) * 0.05, 1e-6)
            ymin -= pad
            ymax += pad

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
    ax.set_ylabel(f's{sample}')

title = f'Multi-bkps frobenius\nwindow_length={best_window_length}'
if best_penalty is not None:
    title += f', penalty={best_penalty:g}'
axes[0].set_title(title)
axes[-1].set_xlabel('snapshot index')
fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, format='pdf', dpi=300, bbox_inches='tight')
print(OUTPUT_PATH)
plt.close(fig)
