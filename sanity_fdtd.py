from coupler.fdtd import FDTDConfig
cfg=FDTDConfig(L_straight_um=10.0)
print(cfg.Nx, cfg.Nz)
print(round(cfg.dt*1e15,3), round(cfg.courant,4))
