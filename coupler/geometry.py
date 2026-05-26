r"""
Cross-section geometry for the symmetric directional coupler.

iSiPP50G layer stack (bottom to top):

    Si wafer (not modelled — too far below to be seen by the mode)
    --------------------------------------------------------------
    Buried oxide (BOX), ~2 um of SiO2
    --------------------------------------------------------------
         __                              __
        /  \                            /  \          <- Si strip, t = 214 nm
       /____\          gap g           /____\
    --------------------------------------------------------------
    Air or SiO2 top cladding


We model everything below the Si strip as semi-infinite SiO2 within the
simulation window. This works because the BOX is ~2 um thick — much greater
than the evanescent penetration depth of the mode (~100 - 200 nm at 1.55 um),
so the bulk Si wafer underneath is optically invisible.

Each Si strip is a symmetric trapezoid. The lithographic sidewall tilts
inward at angle alpha measured from the substrate surface; for the IMEC
process alpha > 85 deg, and we use 85 deg as the worst case. This makes the
strip wider at its base (touching the BOX) than at its top, by

    delta_w = 2 * t / tan(alpha)

so the top width is  w_top = w - delta_w.

Coordinate convention:
    x = transverse axis;  x = 0 at the symmetry plane between the strips
    y = vertical axis;    y = 0 at the bottom face of the Si strips
    z = propagation direction (out of the page; implicit in this 2-D view)
"""



from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .materials import n_si, n_sio2


@dataclass
class CouplerCrossSection:
    """Geometric and material parameters. Defaults follow iSiPP50G."""

    w_nm: float                                   # base width of each Si strip
    D_nm: float = 700.0                           # centre-to-centre spacing
    t_nm: float = 214.0                           # Si layer thickness
    alpha_deg: float = 85.0                       # sidewall angle from substrate
    cladding: Literal["air", "sio2"] = "air"      # top cladding choice
    lam_um: float = 1.55                          # design wavelength

    # Simulation window padding (nm) and grid resolution (nm)
    pad_x_nm: float = 1000.0                      # transverse padding either side
    pad_top_nm: float = 600.0                     # cladding depth above strips
    pad_bot_nm: float = 400.0                     # BOX depth below strips
    dx_nm: float = 10.0
    dy_nm: float = 10.0

    @property
    def w_top_nm(self) -> float:
        """Width of the top face: w - 2 * t / tan(alpha)."""
        return self.w_nm - 2.0 * self.t_nm / np.tan(np.deg2rad(self.alpha_deg))

    @property
    def gap_base_nm(self) -> float:
        """Base gap between the two strips at y = 0:  g = D - w."""
        return self.D_nm - self.w_nm

    @property
    def n_core(self) -> float:
        return float(n_si(self.lam_um))

    @property
    def n_box(self) -> float:
        return float(n_sio2(self.lam_um))

    @property
    def n_cladding(self) -> float:
        return float(n_sio2(self.lam_um)) if self.cladding == "sio2" else 1.0


def _trapezoid_mask(X, Y, xc, w_base, w_top, t):
    """Boolean mask of grid points inside a centred symmetric trapezoid.

    The trapezoid lies in 0 <= y <= t, centred at x = xc. Its half-width
    interpolates linearly from w_base/2 at y = 0 to w_top/2 at y = t.
    """
    half_w = 0.5 * (w_base + (w_top - w_base) * (Y / t))
    return (Y >= 0.0) & (Y <= t) & (np.abs(X - xc) <= half_w)


def eps_function(cs: CouplerCrossSection) -> Callable:
    """Return a closure eps_func(xc_um, yc_um) -> eps array.

    EMpy expects this signature: pass two 1-D arrays of cell-centre coords
    in micrometres, get back a 2-D eps array of shape (len(xc), len(yc)).

    All geometry constants are baked into the closure once so the inner
    callback only depends on the coordinate arrays.
    """
    w_um = cs.w_nm * 1e-3
    wt_um = cs.w_top_nm * 1e-3
    t_um = cs.t_nm * 1e-3
    D_um = cs.D_nm * 1e-3
    n_core = cs.n_core
    n_box = cs.n_box
    n_clad = cs.n_cladding

    def epsfunc(xc_um, yc_um):
        X, Y = np.meshgrid(xc_um, yc_um, indexing="ij")
        # Default: BOX below the strip plane, cladding above
        n = np.where(Y < 0.0, n_box, n_clad)
        # Two Si strips
        for xc in (-D_um / 2.0, +D_um / 2.0):
            #placing the centres of the waveguides
            mask = _trapezoid_mask(X, Y, xc, w_um, wt_um, t_um)
            n = np.where(mask, n_core, n)
        return n ** 2

    return epsfunc


def grid_lines(cs: CouplerCrossSection):
    """Grid-line coordinates in micrometres (length N+1 for N cells).

    The window extends:
        x in [-(D/2 + w/2 + pad_x),  +(D/2 + w/2 + pad_x)]
        y in [-pad_bot,                t + pad_top]
    """
    x_outer = cs.D_nm / 2.0 + cs.w_nm / 2.0 + cs.pad_x_nm
    y_min = -cs.pad_bot_nm
    y_max = cs.t_nm + cs.pad_top_nm
    x_um = np.arange(-x_outer, x_outer + cs.dx_nm * 0.5, cs.dx_nm) * 1e-3
    y_um = np.arange(y_min, y_max + cs.dy_nm * 0.5, cs.dy_nm) * 1e-3
    return x_um, y_um