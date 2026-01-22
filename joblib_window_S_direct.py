# joblib_window_S_direct.py

from joblib import Parallel, delayed
import numpy as np
import pickle
from pathlib import Path

import compute_S_rate
from TemporalNetwork import ContTempNetwork  # only needed if you ever load via ContTempNetwork.load


# Load network 

with open("block1activity.pkl", "rb") as handle:
    net = pickle.load(handle)[0]["tnet"]


# Path
net_id = 0  # change if you run multiple nets
base = Path(f"/scratch/tmp/180/skoove/block1activity/net{net_id}")

# Output dir for entropy results
window = 5
outdir = base / "window_S" / str(window)
outdir.mkdir(parents=True, exist_ok=True)


# Precompute how many windows are needed
considered_times = net.times[net.times < net.times[-1] - window]
num_windows = len(considered_times)


# Worker
def worker(lamda: float):
    try:
        # 1) inter-event matrices for this lamda (needed for window products)
        #    (Assumes laplacians are already available or computable)
        #    If your net already has laplacians computed and stored, you can skip compute_laplacian_matrices.
        if not hasattr(net, "L"):
            net.compute_laplacian_matrices(
                t_start=net.times[0],
                t_stop=net.times[-1],
                random_walk=False,
            )

        net.compute_inter_transition_matrices(
            lamda=lamda,
            t_start=net.times[0],
            t_stop=net.times[-1],
            dense_expm=False,
            use_sparse_stoch=False,
            random_walk=False,
        )

        p0 = np.ones(net.num_nodes) / net.num_nodes
        S_vals = [0.0]  # convention initial value equal to 0
    
        considered_k = np.where(net.times < net.times[-1] - window)[0]
        k0 = considered_k[0]
        def k_to_idx(k):  # store at index 1..num_windows
            return (k - k0) + 1
    
        on_T = compute_S_rate.make_on_window_matrix_entropy_callback_prealloc(p0, S_vals, k_to_idx)

        net.compute_transition_matrices_sliding_timewindow(
            lamda=lamda,
            reverse_time=False,
            window_timelength=window,
            save_intermediate=False,    # <-- crucial: don't store matrices
            on_window_matrix=on_T,      # <-- compute/store entropy scalars
            force_csr=True,
            # tol=...,  # add if used
        )

        #on_T = compute_S_rate.make_on_window_matrix_entropy_callback(p0, S_vals)
            
            # (Optional) sanity: make sure length matches what you expect
            # Expected: 1 + num_windows (because of the leading 0)
            # If TemporalNetwork loop range differs, can remove this.
            # assert len(S_vals) == 1 + num_windows

        S_rate = {"lamda": f"{lamda:.11f}", "window_S": S_vals}

        outfile = outdir / f"window_S{lamda:.11f}"
        with open(outfile, "wb") as f:
            pickle.dump(S_rate, f)

    except Exception as e:
        print(f"error with lamda={lamda:.11f}: {type(e).__name__}: {e}")
        S_rate = {"lamda": f"{lamda:.11f}", "window_S": 10}
        outfile = outdir / f"window_S{lamda:.11f}"
        with open(outfile, "wb") as f:
            pickle.dump(S_rate, f)

    print(lamda)


# Run
lambdas = np.logspace(-5, 0, 10)
n_cpu = 10
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)