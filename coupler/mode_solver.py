"""
Supermode solver for the symmetric directional coupler.

We compute the two lowest-order TE-like supermodes of the cross-section
described by `CouplerCrossSection`, using EMpy's full-vectorial
finite-difference mode solver (VFDModeSolver).

From the supermode effective-index splitting  Δn = n_s − n_a,  the coupled-
mode coupling per unit length follows as

    κ' = π · Δn / λ

This κ' is exactly what the paper plots in Fig. 2(a) and what feeds the
Eq. (1) cross-port power model in `coupling.py`.
"""

from dataclasses import dataclass

import numpy as np
from EMpy.modesolvers.FD import VFDModeSolver

from .geometry import CouplerCrossSection, eps_function, grid_lines


@dataclass
class ModeResult:
    """Pair of supermode effective indices and derived coupling."""

    n_s: float        # symmetric supermode (higher n_eff)
    n_a: float        # anti-symmetric supermode (lower n_eff)
    lam_um: float     # wavelength used, micrometres

    @property
    def delta_n(self) -> float:
        return self.n_s - self.n_a

    @property
    def kappa_per_um(self) -> float:
        """κ' = π · Δn / λ  in rad/um."""
        return np.pi * self.delta_n / self.lam_um


def solve_supermodes(cs: CouplerCrossSection, tol: float = 1e-8) -> ModeResult:
    """Solve for the two lowest supermodes of the coupler cross-section."""
    x_um, y_um = grid_lines(cs)
    eps_fn = eps_function(cs)

    # EMpy boundary spec: "0000" = PEC walls on (W, E, S, N). PEC is fine
    # provided our padding is generous enough that the field has decayed to
    # ~0 before reaching the walls (we set pad_x_nm = 1000 in geometry.py).
    solver = VFDModeSolver(cs.lam_um, x_um, y_um, eps_fn, "0000")
    solver = solver.solve(2, tol)

    n_effs = sorted((m.neff.real for m in solver.modes), reverse=True)
    return ModeResult(n_s=n_effs[0], n_a=n_effs[1], lam_um=cs.lam_um)