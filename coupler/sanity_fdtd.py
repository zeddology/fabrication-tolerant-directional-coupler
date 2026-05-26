from coupler.fdtd import FDTDConfig

cfg = FDTDConfig(L_straight_um=10.0)
print(f"Grid:     {cfg.Nx} x {cfg.Nz} cells")
print(f"dt:       {cfg.dt * 1e15:.3f} fs")
print(f"Courant:  {cfg.courant:.4f}  (must be < 0.707)")
print(f"f0:       {cfg.f0 / 1e12:.2f} THz")
print(f"z_src:    cell {cfg.z_src}")
print(f"z_det:    cell {cfg.z_det}")
print(f"Sim time: {cfg.n_steps * cfg.dt * 1e12:.1f} ps")