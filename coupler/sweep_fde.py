"""
Width sweep using FDE to reproduce Fig 2(a) and Fig 2(b).
Runtime: ~2-3 minutes. Results cached to fde_results.npz after first run.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from coupler.geometry import CouplerCrossSection
from coupler.mode_solver import solve_supermodes
from coupler.coupling import power_coupling

# --- Parameters from the paper ---
widths_nm  = np.arange(460, 561, 5, dtype=float)
D_nm       = 700.0
t_nm       = 214.0
alpha_deg  = 85.0
lam_um     = 1.55
lengths_um = [3.0, 5.0, 10.0]
CACHE      = "fde_results.npz"

# --- Run sweep or load cache ---
if os.path.exists(CACHE):
    print(f"Loading cached results from {CACHE} ...")
    d = np.load(CACHE)
    widths_nm = d["widths_nm"]
    kappa_arr = d["kappa_arr"]
else:
    print(f"Running FDE sweep over {len(widths_nm)} widths ...")
    kappa_arr = np.zeros(len(widths_nm))

    for i, w in enumerate(widths_nm):
        cs = CouplerCrossSection(
            w_nm=float(w), D_nm=D_nm, t_nm=t_nm,
            alpha_deg=alpha_deg, lam_um=lam_um
        )
        res = solve_supermodes(cs)
        kappa_arr[i] = res.kappa_per_um
        print(f"  [{i+1:02d}/{len(widths_nm)}]  "
              f"w={w:.0f} nm   kappa'={kappa_arr[i]:.4f} rad/um")

    np.savez(CACHE, widths_nm=widths_nm, kappa_arr=kappa_arr)
    print(f"Saved to {CACHE}")

# --- Tolerance minimum ---
i_min = int(np.argmin(kappa_arr))
w_star = widths_nm[i_min]
k_star = kappa_arr[i_min]
print(f"\nTolerance minimum:  w* = {w_star:.0f} nm,  kappa' = {k_star:.4f} rad/um")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Fig 2(a)
ax1.plot(widths_nm, kappa_arr, "o-", color="steelblue", lw=1.8, ms=4)
ax1.axvline(w_star, color="tomato", lw=1.0, ls="--", alpha=0.8)
ax1.text(w_star + 2, k_star + 0.0002,
         f"w* = {w_star:.0f} nm", color="tomato", fontsize=9)
ax1.set_xlabel("waveguide width  w  [nm]")
ax1.set_ylabel("coupling per micron  κ'  [rad/μm]")
ax1.set_title("Fig 2(a) — κ'(w) from FDE,  D = 700 nm")
ax1.grid(alpha=0.3)

# Fig 2(b)
colors = ["steelblue", "seagreen", "tomato"]
for L, c in zip(lengths_um, colors):
    P = power_coupling(kappa_arr, L_um=L, kappa0=0.0)
    ax2.plot(widths_nm, P, "o-", color=c, lw=1.8, ms=4, label=f"L = {L:.0f} μm")

    mask = (widths_nm >= w_star - 35) & (widths_nm <= w_star + 35)
    if mask.sum() > 1:
        Ps = P[mask]
        rv = (Ps.max() - Ps.min()) / (Ps.max() + Ps.min()) * 100
        print(f"  L={L:.0f} um:  relative variation in ±35 nm window = {rv:.1f}%")

ax2.axvline(w_star, color="gray", lw=1.0, ls="--", alpha=0.6)
ax2.set_xlabel("waveguide width  w  [nm]")
ax2.set_ylabel("cross-port power coupling  P")
ax2.set_title("Fig 2(b) — P(w),  D = 700 nm")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig2_fde_sweep.png", dpi=150)
plt.show()
print("Plot saved to fig2_fde_sweep.png")