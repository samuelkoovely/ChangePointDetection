"""
Compute the spectrum (eigenvalues) of A1*A2*A3 where Ai = exp(-Li)
for the Laplacian counterexample matrices L1, L2, L3 (4 nodes).
"""

import numpy as np
from scipy.linalg import expm, eigvals


L1 = np.array([
    [1, -1, 0],
    [-1, 1, 0],
    [0, 0, 0],
], dtype=float)

L2 = np.array([
    [ 0, 0,0],
    [ 0, 1,-1],
    [0,-1, 1],
], dtype=float)

L3 = np.array([
    [ 1, 0,-1],
    [ 0, 0, 0],
    [-1, 0, 1],
], dtype=float)

# Heat kernels
A1 = expm(-L1)
A2 = expm(-L2)
A3 = expm(-L3)

# Product (inhomogeneous diffusion)
P = A1 @ A2 @ A3

# Spectrum
lam = eigvals(P)  # complex in general

# Nicely formatted output
print("Eigenvalues of A1*A2*A3:")
# sort by real part then imag part for stable display
lam_sorted = sorted(lam, key=lambda z: (np.real(z), np.imag(z)))
for z in lam_sorted:
    # print small imaginary parts cleanly
    if abs(z.imag) < 1e-12:
        print(f"  {z.real:.12g}")
    else:
        print(f"  {z.real:.12g} {z.imag:+.12g}j")