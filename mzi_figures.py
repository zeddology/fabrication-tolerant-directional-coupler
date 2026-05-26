"""
Reproduce Fig 3 and Fig 4 from Liu/Khan/Bogaerts (ECIO 2023):
    Fig 3 - example MZI bar/cross spectrum for a single width.
    Fig 4 - P(w) ground truth (from FDE kappa) vs P fitted from a noisy
            synthetic MZI bar spectrum.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from coupler.coupling import power_coupling, mzi_outputs


# --- Cached FDE results ---
data       = np.load("fde_results.npz")
widths_nm  = data["widths_nm"]
kappa_arr  = data["kappa_arr"]

# Smooth the FDE kappa(w) with a cubic polyfit to suppress grid noise
coeffs       = np.polyfit(widths_nm, kappa_arr, 3)
kappa_smooth = np.polyval(coeffs, widths_nm)

print("--- kappa_arr (rad/um) ---")
print("width   raw       smooth     diff")
for w, k_raw, k_sm in zip(widths_nm, kappa_arr, kappa_smooth):
    print(f"{w:5.0f}   {k_raw:.6f}  {k_sm:.6f}  {k_sm - k_raw:+.6f}")
print()

# --- DC layout (Fig 4 conditions in the paper) ---
L      = 9.98    # um, straight coupling length
kappa0 = 0.5     # rad, lumped bend contribution

P_fde = power_coupling(kappa_smooth, L, kappa0)

# --- MZI parameters ---
n_eff = 2.4
dL    = 150.0    # um
lam   = np.linspace(1.520, 1.580, 4000)


# ============================================================
# FIG 3 - single MZI spectrum at the width closest to P = 0.5
# ============================================================
i_pick  = int(np.argmin(np.abs(P_fde - 0.5)))
w_pick  = float(widths_nm[i_pick])
P_pick  = float(P_fde[i_pick])

P_bar, P_cross = mzi_outputs(P_pick, n_eff, dL, lam)

# Floor for log to keep notches from going to -inf
EPS = 1e-10
P_bar_dB   = 10.0 * np.log10(np.maximum(P_bar,   EPS))
P_cross_dB = 10.0 * np.log10(np.maximum(P_cross, EPS))

plt.figure(figsize=(10, 5))
plt.plot(lam * 1000, P_cross_dB, label="cross", lw=1.0)
plt.plot(lam * 1000, P_bar_dB,   label="bar",   lw=1.0)
plt.ylim(-50, -5)
plt.xlabel("wavelength [nm]")
plt.ylabel("power [dB]")
plt.title(f"Fig 3 - MZI spectrum, w={w_pick:g} nm  (P={P_pick:.2f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_mzi_spectrum.png", dpi=150)
plt.close()


# ============================================================
# FIG 4 - P(w) recovered from synthetic MZI bar-spectra fits
# ============================================================
rng = np.random.default_rng(42)
NOISE_dB = 0.05
print(f"--- NOISE_dB in use: {NOISE_dB} ---\n")

fitted_P = np.zeros_like(widths_nm)

def fit_P_from_bar(bar_dB):
    """Least-squares fit of cross-port coupling P from a noisy bar spectrum."""
    def cost(P_trial):
        b, _ = mzi_outputs(P_trial, n_eff, dL, lam)
        b_dB = 10.0 * np.log10(np.maximum(b, EPS))
        return float(np.sum((b_dB - bar_dB) ** 2))
    res = minimize_scalar(cost, bounds=(0.01, 0.99), method="bounded")
    return float(res.x)

for i, w in enumerate(widths_nm):
    P_true = P_fde[i]
    bar_clean, _ = mzi_outputs(P_true, n_eff, dL, lam)
    bar_dB_clean = 10.0 * np.log10(np.maximum(bar_clean, EPS))
    bar_dB_noisy = bar_dB_clean + rng.normal(0.0, NOISE_dB, size=lam.shape)
    fitted_P[i] = fit_P_from_bar(bar_dB_noisy)

# Tolerance: minimum of |dP / dw|  (=  most width-tolerant operating point)
dP_dw       = np.gradient(P_fde, widths_nm)
i_tol_min   = int(np.argmin(np.abs(dP_dw)))
w_tol_min   = float(widths_nm[i_tol_min])

plt.figure(figsize=(10, 5))
plt.plot(widths_nm, P_fde,    "-",  label="P(w) - FDE truth", lw=1.5)
plt.plot(widths_nm, fitted_P, "o",  label="P fitted from MZI", ms=6, mfc="none")
plt.axvline(w_tol_min, color="red", ls="--",
            label=f"tolerance opt: min |dP/dw| @ w={w_tol_min:g} nm")
plt.xlabel("waveguide width [nm]")
plt.ylabel("cross-port power coupling P")
plt.title("Fig 4 - P(w) from MZI fit, L=9.98um")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_P_vs_width.png", dpi=150)
plt.close()

# --- 5-point truth vs fitted comparison (evenly spaced widths) ---
sample_idx = np.linspace(0, len(widths_nm) - 1, 5, dtype=int)
print("--- 5-point truth vs fit comparison ---")
print("width   P_true    P_fitted   diff")
for i in sample_idx:
    print(f"{widths_nm[i]:5.0f}   {P_fde[i]:.4f}    {fitted_P[i]:.4f}    {fitted_P[i] - P_fde[i]:+.4f}")
print()

# --- Report ---
residuals  = fitted_P - P_fde
rmse       = float(np.sqrt(np.mean(residuals ** 2)))
max_err    = float(np.max(np.abs(residuals)))
mean_bias  = float(np.mean(residuals))

print("=" * 60)
print(f"Fig 3 pick:   w = {w_pick:g} nm  (P = {P_pick:.3f})")
print(f"Saved: fig3_mzi_spectrum.png")
print()
print(f"Tolerance minimum (min |dP/dw|):")
print(f"  width = {w_tol_min:g} nm")
print(f"  P     = {P_fde[i_tol_min]:.4f}")
print(f"  |dP/dw| = {abs(dP_dw[i_tol_min]):.4e} per nm")
print()
print(f"Fit quality (truth vs MZI-fitted P over {len(widths_nm)} widths):")
print(f"  RMSE         = {rmse:.4f}")
print(f"  max |error|  = {max_err:.4f}")
print(f"  mean bias    = {mean_bias:+.4f}   (positive = fits sit above truth)")
print(f"Saved: fig4_P_vs_width.png")
print("=" * 60)
