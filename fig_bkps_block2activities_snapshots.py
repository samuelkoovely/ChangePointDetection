import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('gridsearch_results/block2activities_snapshots/gridsearch_results.pkl', 'rb') as handle:
    results_full = pickle.load(handle)

with open('gridsearch_results/block2activities_snapshots_lin_approx/gridsearch_results.pkl', 'rb') as handle:
    results_lin = pickle.load(handle)

def get_best_signal_metadata(results):
    lambdas = results['lambdas']
    windows = results['windows']
    lamda = lambdas[results['best_index'][0]]
    window = windows[results['best_index'][1]]
    predicted_change_points = results['lambda_results'][results['best_index'][0]]['predicted_change_points'][window]
    return lamda, window, predicted_change_points

lamda_full, window_full, predicted_change_points_full = get_best_signal_metadata(results_full)
lamda_lin, window_lin, predicted_change_points_lin = get_best_signal_metadata(results_lin)

fig, axes = plt.subplots(5, 2, figsize=(18, 10), sharex=False, squeeze=False)
for sample in range(5):
    with open('data/block2activities_snapshots.pkl', 'rb') as handle:
        tnet = pickle.load(handle)[sample]
        bkp = float(tnet['bkp'])

    with open(
        f'gridsearch_results/block2activities_snapshots/signals/sample_{sample}/'
        f'signal_lamda_{lamda_full:.11f}_window_{window_full:g}.pkl',
        'rb',
    ) as handle:
        signal_full = pickle.load(handle)

    with open(
        f'gridsearch_results/block2activities_snapshots_lin_approx/signals/sample_{sample}/'
        f'signal_lamda_{lamda_lin:.11f}_window_{window_lin:g}.pkl',
        'rb',
    ) as handle:
        signal_lin = pickle.load(handle)

    ax_full = axes[sample, 0]
    ax_full.plot(signal_full['signal'], label=f'sample_{sample}')
    ymin_full = np.min(signal_full['signal'])
    ymax_full = np.max(signal_full['signal'])
    ax_full.vlines(
        tnet['bkp'],
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
    ax_full.legend(loc='upper right')
    ax_full.set_ylabel(f's{sample}')

    ax_lin = axes[sample, 1]
    ax_lin.plot(signal_lin['signal'], label=f'sample_{sample}')
    ymin_lin = np.min(signal_lin['signal'])
    ymax_lin = np.max(signal_lin['signal'])
    ax_lin.vlines(
        tnet['bkp'],
        ymin=ymin_lin,
        ymax=ymax_lin,
        color='black',
    )
    for pred_cp in predicted_change_points_lin[sample]:
        ax_lin.vlines(
            pred_cp,
            ymin=ymin_lin,
            ymax=ymax_lin,
            linestyles='dashed',
            color='red',
        )
    ax_lin.legend(loc='upper right')

axes[0, 0].set_title(
    f'Original inter-T\nlambda={lamda_full:.5g}, window={window_full:g}'
)
axes[0, 1].set_title(
    f'Linear approximation\nlambda={lamda_lin:.5g}, window={window_lin:g}'
)
axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')
fig.tight_layout()
#plt.savefig('/home/b/skoove/Desktop/ChangePointDetection/fig_bkps_block2activities_snapshots.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
