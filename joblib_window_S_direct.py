# joblib_window_S_direct.py

from joblib import Parallel, delayed
import numpy as np
import pickle
from pathlib import Path

import time

import compute_S_rate


# Load network 

with open("block1activity.pkl", "rb") as handle:
    net = pickle.load(handle)[0]["tnet"]


# Path
net_id = 0  # change if you run multiple nets
base = Path(f"/scratch/tmp/180/skoove/block1activity/net{net_id}")

# Window scan configuration (2D scan over (lamda, window))
# Choose whatever range you want; this is a sensible default.
windows = [1, 5, 10, 25]

# Sampling configuration (option 3: uniform in time)
sample_fraction = 0.1  # e.g. 0.1 = compute ~10% of windows per window-length


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

        # For each window length, sample ks uniformly in time and compute entropy only at those ks (option 3 + 5A)
        for window in windows:
            # Windows start times that are valid for the chosen window length
            considered_times = net.times[net.times < net.times[-1] - window]
            M = len(considered_times)
            if M <= 0:
                # No valid windows for this window length
                continue

            m = max(1, int(np.ceil(sample_fraction * M)))

            # Targets uniformly spaced in time, then mapped to closest k via searchsorted
            t_targets = np.linspace(float(considered_times[0]), float(considered_times[-1]), m)
            k_samples = np.searchsorted(considered_times, t_targets, side="left")
            k_samples = np.clip(k_samples, 0, M - 1)
            k_samples = np.unique(k_samples).astype(int)

            # Convenience: sampled times (useful to store alongside entropy)
            t_samples = net.times[k_samples]

            # Preallocate only for sampled points (option 5A)
            S_arr = np.empty(len(k_samples), dtype=float)
            k_to_pos = {int(k): i for i, k in enumerate(k_samples)}

            def k_to_idx(k):
                return k_to_pos[int(k)]

            on_T = compute_S_rate.make_on_window_matrix_entropy_callback_prealloc(p0, S_arr, k_to_idx)

            net.compute_transition_matrices_sliding_timewindow(
                lamda=lamda,
                reverse_time=False,
                window_timelength=window,
                save_intermediate=False,    # <-- crucial: don't store matrices
                on_window_matrix=on_T,      # <-- compute/store entropy scalars
                force_csr=True,
                k_samples=k_samples,
                # tol=...,  # add if used
            )

            S_rate = {
                "lamda": f"{lamda:.11f}",
                "window": window,
                "k_samples": k_samples,
                "t_samples": t_samples,
                "signal": S_arr,
            }

            outdir = base / "window_S" / str(window)
            outdir.mkdir(parents=True, exist_ok=True)

            outfile = outdir / f"window_S{lamda:.11f}"
            with open(outfile, "wb") as f:
                pickle.dump(S_rate, f)

    except Exception as e:
        print(f"error with lamda={lamda:.11f}: {type(e).__name__}: {e}")
        for window in windows:
            outdir = base / "window_S" / str(window)
            outdir.mkdir(parents=True, exist_ok=True)
            S_rate = {"lamda": f"{lamda:.11f}", "window": window, "signal": 10}
            outfile = outdir / f"window_S{lamda:.11f}"
            with open(outfile, "wb") as f:
                pickle.dump(S_rate, f)

    print(lamda)


# Run
lambdas = np.logspace(-5, 0, 10)
n_cpu = 10

t_start = time.time()
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)
t_end = time.time()

elapsed = t_end - t_start
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"Total runtime: {elapsed:.2f} s ({int(h):02d}:{int(m):02d}:{s:05.2f})")
