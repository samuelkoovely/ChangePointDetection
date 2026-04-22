import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def resolve_existing_path(path_str: str | Path, base_dir: Path) -> Path:
    path = Path(path_str)
    candidates = [path, base_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_dir / path


def get_results_path() -> Path:
    return (
        BASE_DIR
        / 'gridsearch_results/block1activity_snapshots_frobenius/gridsearch_results.pkl'
    )


def get_signal_result_candidates(
    signals_outdir: Path,
    sample_dir: str,
    window_length: int,
) -> list[Path]:
    return [
        signals_outdir
        / sample_dir
        / f'signal_window_length_{int(window_length)}.pkl',
        signals_outdir
        / sample_dir
        / f'signal_window_{float(window_length):g}.pkl',
    ]


def get_signal_result_path(
    signals_outdir: Path,
    sample_dir: str,
    window_length: int,
) -> Path:
    candidates = get_signal_result_candidates(
        signals_outdir=signals_outdir,
        sample_dir=sample_dir,
        window_length=window_length,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


results_path = get_results_path()
with open(results_path, 'rb') as handle:
    results = pickle.load(handle)

best_window_length = results.get('best_window_length')
if best_window_length is None:
    best_index = results.get('best_index')
    if best_index is None:
        raise ValueError('No best window_length found in grid-search results.')
    best_window_length = results['window_lengths'][best_index]

signals_outdir = results.get('signals_outdir')
if signals_outdir is None:
    signals_outdir = results_path.parent / 'signals'
else:
    signals_outdir = resolve_existing_path(signals_outdir, BASE_DIR)

best_result = results.get('window_results')
if best_result is None:
    raise ValueError('No window_results found in Frobenius grid-search results.')

predicted_change_points = best_result['predicted_change_points'][int(best_window_length)]
sample_names = best_result.get('sample_names')
if sample_names is None:
    sample_names = [f'sample_{i}' for i in range(len(predicted_change_points))]

with open(BASE_DIR / 'data/block1activity_train_snapshots.pkl', 'rb') as handle:
    dataset = pickle.load(handle)

n_samples = min(5, len(dataset), len(predicted_change_points))
fig, axes = plt.subplots(n_samples, 1, figsize=(14, 8), sharex=False)
axes = np.atleast_1d(axes)

for sample in range(n_samples):
    entry = dataset[sample]
    sample_name = sample_names[sample] if sample_names[sample] is not None else f'sample_{sample}'

    signal_path = get_signal_result_path(
        signals_outdir=signals_outdir,
        sample_dir=sample_name,
        window_length=int(best_window_length),
    )
    with open(signal_path, 'rb') as handle:
        signal = pickle.load(handle)

    signal_values = np.asarray(signal['signal'])
    x_values = np.asarray(signal.get('snapshot_indices', np.arange(len(signal_values))))

    ax = axes[sample]
    ax.plot(x_values, signal_values, label=f'sample_{sample}')
    if signal_values.size > 0:
        ymin = np.min(signal_values)
        ymax = np.max(signal_values)
        for bkp in entry['bkps']:
            ax.vlines(
                bkp,
                ymin=ymin,
                ymax=ymax,
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

axes[0].set_title(f'Frobenius distance, window_length={int(best_window_length)}')
axes[-1].set_xlabel('time')
fig.tight_layout()
plt.show()
