import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / 'figures' / f'{Path(__file__).stem}.pdf'

with open(BASE_DIR / 'gridsearch_results/block1activity_snapshots/gridsearch_results.pkl', 'rb') as handle:
    results = pickle.load(handle)

with open(BASE_DIR / 'data/block1activity_train_snapshots.pkl', 'rb') as handle:
    dataset = pickle.load(handle)

lambdas = results['lambdas']
windows = results['windows']
lamda = lambdas[results['best_index'][0]]
window = windows[results['best_index'][1]]

predicted_change_points = results['lambda_results'][results['best_index'][0]]['predicted_change_points'][window]
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
    tnet = dataset[sample]
    bkps = [float(change_point) for change_point in tnet['bkps']]

    with open(
        BASE_DIR
        / 'gridsearch_results/block1activity_snapshots/signals'
        / f'sample_{sample}'
        / f'signal_lamda_{lamda:.11f}_window_{window:g}.pkl',
        'rb',
    ) as handle:
        signal = pickle.load(handle)

    ax = axes[sample]
    signal_values = np.asarray(signal['signal'], dtype=float)
    x_values = np.asarray(signal.get('snapshot_indices', signal.get('k_samples', np.arange(len(signal_values)))), dtype=float)
    ax.plot(x_values, signal_values, color='C0')
    ymin = np.min(signal_values)
    ymax = np.max(signal_values)
    if np.isclose(ymin, ymax):
        pad = max(abs(float(ymin)) * 0.05, 1e-6)
        ymin -= pad
        ymax += pad
    for bkp in bkps:
        ax.vlines(
            bkp,
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
            color='red'
        )
    ax.set_ylabel(f's{sample}')
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
