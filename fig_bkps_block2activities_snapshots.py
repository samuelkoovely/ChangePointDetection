import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('gridsearch_results/block2activities_snapshots/gridsearch_results.pkl', 'rb') as handle:
    results = pickle.load(handle)

lambdas = results['lambdas']
windows = results['windows']
lamda = lambdas[results['best_index'][0]]
window = windows[results['best_index'][1]]

predicted_change_points = results['lambda_results'][results['best_index'][0]]['predicted_change_points'][window]
fig, axes = plt.subplots(5, 1, figsize=(14, 8), sharex=False)
for sample in range(5):
    with open('block2activities_snapshots.pkl', 'rb') as handle:
        tnet = pickle.load(handle)[sample]
        bkp = float(tnet['bkp'])

    with open(f'gridsearch_results/block2activities_snapshots/signals/sample_{sample}/signal_lamda_{lamda:.11f}_window_{window:g}.pkl', 'rb') as handle:
        signal = pickle.load(handle)

    ax = axes[sample]
    ax.plot(signal['signal'], label=f'sample_{sample}')
    ymin = np.min(signal['signal'])
    ymax = np.max(signal['signal'])
    ax.vlines(
        tnet['bkp'],
        ymin=ymin,
        ymax=ymax
    )
    for pred_cp in predicted_change_points[sample]:
        ax.vlines(
            pred_cp,
            ymin=ymin,
            ymax=ymax,
            linestyles='dashed',
            color='red'
        )
    ax.legend(loc='upper right')
    ax.set_ylabel(f's{sample}')
axes[-1].set_xlabel('time')
fig.tight_layout()
plt.show()
