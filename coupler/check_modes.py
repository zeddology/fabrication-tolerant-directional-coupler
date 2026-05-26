from coupler.geometry import CouplerCrossSection
from coupler.mode_solver import solve_supermodes

cs = CouplerCrossSection(w_nm=450.0)
res = solve_supermodes(cs)

print(f"n_s  = {res.n_s:.5f}")
print(f"n_a  = {res.n_a:.5f}")
print(f"Δn   = {res.delta_n:.5f}")
print(f"κ'   = {res.kappa_per_um:.4f}  rad/um")
print(f"L_π  = {3.14159 / (2 * res.kappa_per_um):.1f}  um")