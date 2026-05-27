# joblib_window_S_direct.py

from joblib import Parallel, delayed
import numpy as np
import pickle
from pathlib import Path

import time

from signal_generation import compute_signals_for_lambda


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
reverse_time = False


# Worker
def worker(lamda: float):
    try:
        signals_by_window = compute_signals_for_lambda(
            net=net,
            lamda=lamda,
            windows=windows,
            sample_fraction=sample_fraction,
            window_backend="segment_tree",
            reverse_time=reverse_time,
        )

        subdir = "window_S_rev" if reverse_time else "window_S"
        for window in windows:
            signal_result = signals_by_window[float(window)]
            outdir = base / subdir / str(window)
            outdir.mkdir(parents=True, exist_ok=True)

            outfile = outdir / f"window_S{lamda:.11f}"
            with open(outfile, "wb") as f:
                pickle.dump(signal_result, f)

    except Exception as e:
        print(f"error with lamda={lamda:.11f}: {type(e).__name__}: {e}")
        subdir = "window_S_rev" if reverse_time else "window_S"
        for window in windows:
            outdir = base / subdir / str(window)
            outdir.mkdir(parents=True, exist_ok=True)
            S_rate = {
                "lamda": float(lamda),
                "window": float(window),
                "signal": 10,
                "reverse_time": bool(reverse_time),
                "direction": "backward" if reverse_time else "forward",
            }
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
