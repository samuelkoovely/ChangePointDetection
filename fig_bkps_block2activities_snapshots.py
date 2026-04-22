import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / 'figures' / f'{Path(__file__).stem}.pdf'

with open(BASE_DIR / 'gridsearch_results/block2activities_snapshots/gridsearch_results.pkl', 'rb') as handle:
    results_full = pickle.load(handle)

with open(BASE_DIR / 'data/block2activities_train_snapshots.pkl', 'rb') as handle:
    dataset = pickle.load(handle)

def get_best_signal_metadata(results):
    lambdas = results['lambdas']
    windows = results['windows']
    lamda = lambdas[results['best_index'][0]]
    window = windows[results['best_index'][1]]
    predicted_change_points = results['lambda_results'][results['best_index'][0]]['predicted_change_points'][window]
    return lamda, window, predicted_change_points

lamda_full, window_full, predicted_change_points_full = get_best_signal_metadata(results_full)

n_samples = min(len(dataset), len(predicted_change_points_full))
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
    tnet = dataset[sample]
    bkps = [float(change_point) for change_point in tnet['bkps']]

    with open(
        BASE_DIR
        / 'gridsearch_results/block2activities_snapshots/signals'
        / f'sample_{sample}'
        / f'signal_lamda_{lamda_full:.11f}_window_{window_full:g}.pkl',
        'rb',
    ) as handle:
        signal_full = pickle.load(handle)

    ax_full = axes[sample]
    signal_values = np.asarray(signal_full['signal'], dtype=float)
    x_values = np.asarray(signal_full.get('snapshot_indices', signal_full.get('k_samples', np.arange(len(signal_values)))), dtype=float)
    ax_full.plot(x_values, signal_values, color='C0')
    ymin_full = np.min(signal_values)
    ymax_full = np.max(signal_values)
    if np.isclose(ymin_full, ymax_full):
        pad = max(abs(float(ymin_full)) * 0.05, 1e-6)
        ymin_full -= pad
        ymax_full += pad
    for bkp in bkps:
        ax_full.vlines(
            bkp,
            ymin=ymin_full,
            ymax=ymax_full,
            color='black',
        )
    for pred_cp in predicted_change_points_full[sample]:
        ax_full.vlines(
            pred_cp,
            ymin=ymin_full,
            ymax=ymax_full,
            linestyles='dashed',
            color='red',
        )
    ax_full.set_ylabel(f's{sample}')

axes[0].set_title(
    f'Original inter-T\nlambda={lamda_full:.5g}, window={window_full:g}'
)
axes[-1].set_xlabel('time')
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
