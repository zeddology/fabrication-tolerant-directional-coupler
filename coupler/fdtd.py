r"""
2D TE FDTD simulation of the directional coupler.

Uses the effective-index method to collapse the 3D cross-section into a 2D
top-down (x, z) simulation plane. x is the transverse axis, z is propagation.

Field components solved (TE polarisation):
    Ex(i, k)  — transverse electric field
    Ez(i, k)  — longitudinal electric field
    Hy(i, k)  — out-of-plane magnetic field

Yee leapfrog update equations:
    Hy ← Hy + (dt / mu0) * ( dEx/dz  −  dEz/dx )
    Ex ← Ex + (dt / eps0 / eps) *  dHy/dz
    Ez ← Ez − (dt / eps0 / eps) *  dHy/dx

Absorbing boundaries : CPML on all four sides.
Source               : soft CW sinusoid injected at the input port plane.
Power extraction     : time-averaged Poynting flux at output port planes.
"""

import numpy as np
from dataclasses import dataclass

# ── Physical constants (SI) ───────────────────────────────────────────────────

C0   = 299_792_458.0           # speed of light,         m/s
MU0  = 4e-7 * np.pi            # free-space permeability, H/m
EPS0 = 1.0 / (MU0 * C0 ** 2)  # free-space permittivity, F/m


# ── Simulation configuration ──────────────────────────────────────────────────

@dataclass
class FDTDConfig:
    """All parameters needed to define and run one FDTD simulation.

    Spatial units: micrometres (um) for human-readable input.
    Time units:    seconds internally (SI).
    """

    # --- Coupler geometry ----------------------------------------------------
    L_straight_um: float          # straight coupling section length [um]
    D_um:          float = 0.700  # centre-to-centre waveguide spacing [um]
    w_um:          float = 0.450  # waveguide width [um]
    L_bend_um:     float = 8.0    # length of each S-bend [um] (0 = no bends)
    sep_in_um:     float = 4.0    # waveguide separation at input/output ports [um]

    # --- Effective indices (from mode_solver.py) -----------------------------
    n_eff_strip: float = 2.38    # TE n_eff of a single Si strip
    n_eff_clad:  float = 1.444   # effective index outside the strip (BOX region)

    # --- Wavelength ----------------------------------------------------------
    lam0_um: float = 1.55        # centre wavelength [um]

    # --- Grid ----------------------------------------------------------------
    dl_um:     float = 0.050     # spatial step, same in x and z [um]
    n_pml:     int   = 15        # number of CPML cells on each side
    n_pad_src: int   = 40        # extra cells between PML and coupler start (was 20)
    n_pad_det: int   = 20        # extra cells between coupler end and detector

    # --- Time ----------------------------------------------------------------
    n_steps: int = 40_000        # total time steps to run

    # ── Derived quantities (properties) ──────────────────────────────────────

    @property
    def dl(self) -> float:
        """Spatial step in metres."""
        return self.dl_um * 1e-6

    @property
    def dt(self) -> float:
        """Time step in seconds — 99 % of the 2D CFL stability limit.

        CFL condition (2D, equal spacing):
            dt  <=  dl / (c * sqrt(2))
        """
        return 0.99 * self.dl / (C0 * np.sqrt(2))

    @property
    def courant(self) -> float:
        """Courant number C = c * dt / dl. Must be < 1/sqrt(2) ≈ 0.707."""
        return C0 * self.dt / self.dl

    @property
    def lam0(self) -> float:
        """Centre wavelength in metres."""
        return self.lam0_um * 1e-6

    @property
    def f0(self) -> float:
        """Centre frequency in Hz."""
        return C0 / self.lam0

    @property
    def omega0(self) -> float:
        """Angular frequency in rad/s."""
        return 2.0 * np.pi * self.f0

    @property
    def Nz_bend(self) -> int:
        """Grid cells for each S-bend section."""
        return int(round(self.L_bend_um / self.dl_um))

    @property
    def Nz_straight(self) -> int:
        """Grid cells for the straight coupling section."""
        return int(round(self.L_straight_um / self.dl_um))

    @property
    def Nz_core(self) -> int:
        """Total z-cells for the coupler body (bends + straight + bends)."""
        return 2 * self.Nz_bend + self.Nz_straight

    @property
    def Nz(self) -> int:
        """Total z grid size including PML and padding."""
        return self.n_pml + self.n_pad_src + self.Nz_core + self.n_pad_det + self.n_pml

    @property
    def Nx(self) -> int:
        """Total x grid size.

        Wide enough to contain:
            PML | padding | left waveguide | gap/bend region | right waveguide | padding | PML
        Maximum x-extent comes from the input port where waveguides are sep_in_um apart.
        """
        half_span = self.sep_in_um / 2.0 + self.w_um + 1.0   # um, 1 um extra padding
        return self.n_pml + int(round(2.0 * half_span / self.dl_um)) + self.n_pml

    @property
    def x_centre(self) -> int:
        """Grid index of x = 0 (symmetry plane between waveguides)."""
        return self.Nx // 2

    @property
    def z_coupler_start(self) -> int:
        """Grid index where the coupler body (first bend) begins."""
        return self.n_pml + self.n_pad_src

    @property
    def z_src(self) -> int:
        """Source injection plane — 10 cells into the bend so the source sits
        inside a defined waveguide and radiation has the full padding region
        to dissipate into the input-side PML."""
        return self.z_coupler_start + 10

    @property
    def z_det(self) -> int:
        """Grid index of the detector (output) plane."""
        return self.z_coupler_start + self.Nz_core
    
# ── Effective index method ────────────────────────────────────────────────────

def slab_te_neff(n_core, n_sub, n_clad, t_m, lam_m):
    """TE0 effective index of a 3-layer asymmetric slab waveguide.

    Layer stack (bottom to top): substrate / core / cladding.

    Solves the TE dispersion relation for the fundamental (m=0) mode:

        kf * t  =  arctan(gamma_sub / kf)  +  arctan(gamma_clad / kf)

    where:
        kf         = sqrt(k0^2 n_core^2 - beta^2)    transverse wavevector in core
        gamma_sub  = sqrt(beta^2 - k0^2 n_sub^2)     evanescent decay into substrate
        gamma_clad = sqrt(beta^2 - k0^2 n_clad^2)    evanescent decay into cladding
        beta       = k0 * n_eff                       propagation constant

    Parameters
    ----------
    n_core : float  refractive index of the core (Si ~ 3.48)
    n_sub  : float  refractive index of the substrate (SiO2 ~ 1.44)
    n_clad : float  refractive index of the cladding (air = 1.0)
    t_m    : float  core thickness in metres
    lam_m  : float  wavelength in metres

    Returns
    -------
    n_eff : float   effective index of the TE0 slab mode
    """
    from scipy.optimize import brentq

    k0 = 2.0 * np.pi / lam_m
    n_min = max(n_sub, n_clad)   # below this, no guided mode exists
    n_max = n_core               # above this, kf becomes imaginary

    def dispersion(n_eff):
        beta = k0 * n_eff
        kf = np.sqrt(k0 ** 2 * n_core ** 2 - beta ** 2)
        g_sub = np.sqrt(beta ** 2 - k0 ** 2 * n_sub ** 2)
        g_clad = np.sqrt(beta ** 2 - k0 ** 2 * n_clad ** 2)
        # LHS - RHS of the dispersion relation; zero-crossing = guided mode
        return kf * t_m - np.arctan(g_sub / kf) - np.arctan(g_clad / kf)

    n_eff = brentq(dispersion, n_min + 1e-10, n_max - 1e-10)
    return n_eff


def compute_effective_indices(cs):
    """Compute the two effective indices needed for the 2D FDTD layout.

    Parameters
    ----------
    cs : CouplerCrossSection   (from geometry.py)

    Returns
    -------
    n_eff_strip : float   TE0 n_eff of the Si slab (where waveguides exist)
    n_eff_clad  : float   effective index outside the strips (BOX region)
    """
    from .geometry import CouplerCrossSection  # for type clarity only

    n_eff_strip = slab_te_neff(
        n_core=cs.n_core,        # Si ~ 3.48
        n_sub=cs.n_box,           # SiO2 ~ 1.44
        n_clad=cs.n_cladding,     # air = 1.0 (or SiO2 if oxide-clad)
        t_m=cs.t_nm * 1e-9,       # 214 nm → metres
        lam_m=cs.lam_um * 1e-6    # 1.55 um → metres
    )

    # Outside the strip there is no Si core, so no guided slab mode exists.
    # We use the substrate (BOX) index as the background. This is the standard
    # choice for the effective-index method on SOI.
    n_eff_clad = cs.n_box

    return n_eff_strip, n_eff_clad


# ── 2D layout builder ─────────────────────────────────────────────────────────

def build_layout(cfg, n_strip, n_clad):
    """Build the 2D refractive index map for the coupler.

    Two symmetric waveguides run along z. Each follows:
        S-bend in   : centre eases from +/- sep_in/2  -> +/- D/2  (raised cosine)
        Straight    : centre held at +/- D/2
        S-bend out  : mirror of S-bend in, back to +/- sep_in/2

    Parameters
    ----------
    cfg     : FDTDConfig
    n_strip : float   index inside each waveguide stripe
    n_clad  : float   background index everywhere else

    Returns
    -------
    n : ndarray, shape (cfg.Nx, cfg.Nz)
    """
    n = np.full((cfg.Nx, cfg.Nz), n_clad, dtype=float)

    x_um = (np.arange(cfg.Nx) - cfg.x_centre) * cfg.dl_um
    half_w = cfg.w_um / 2.0

    x_in = cfg.sep_in_um / 2.0
    x_str = cfg.D_um / 2.0

    Nz_bend = cfg.Nz_bend
    Nz_straight = cfg.Nz_straight

    for sign in (-1.0, +1.0):
        for k_local in range(cfg.Nz_core):
            if k_local < Nz_bend:
                # S-bend in: raised cosine ease from x_in to x_str
                frac = 0.5 * (1.0 - np.cos(np.pi * k_local / Nz_bend))
                xc = sign * (x_in + (x_str - x_in) * frac)
            elif k_local < Nz_bend + Nz_straight:
                # Straight coupling section
                xc = sign * x_str
            else:
                # S-bend out: mirror of S-bend in
                k_out = k_local - Nz_bend - Nz_straight
                frac = 0.5 * (1.0 - np.cos(np.pi * k_out / Nz_bend))
                xc = sign * (x_str + (x_in - x_str) * frac)

            k = cfg.z_coupler_start + k_local
            mask = np.abs(x_um - xc) <= half_w
            n[mask, k] = n_strip

    return n


# ── CPML conductivity profiles ────────────────────────────────────────────────

def pml_coefficients(cfg, n_eff_strip=2.81):
    """1D conductivity profiles for the CPML absorbing boundaries.

    Cubic polynomial grading, sigma(d) = sigma_max * (d / n_pml)**3, where d
    is the depth into the PML in cells (1 .. n_pml). sigma_max is set for a
    target round-trip reflection R = 1e-8:

        sigma_max = -3 * c0 * ln(R) / (2 * n_eff * n_pml * dl)

    Parameters
    ----------
    cfg          : FDTDConfig
    n_eff_strip  : float   reference index used in sigma_max (default 2.81)

    Returns
    -------
    sigma_x : ndarray, shape (cfg.Nx,)
    sigma_z : ndarray, shape (cfg.Nz,)
    """
    R = 1e-8
    sigma_max = -(3.0 * C0 * EPS0 * np.log(R)) / (2.0 * n_eff_strip * cfg.n_pml * cfg.dl)

    d = np.arange(1, cfg.n_pml + 1)
    profile = sigma_max * (d / cfg.n_pml) ** 3   # increases into the PML

    sigma_x = np.zeros(cfg.Nx)
    sigma_z = np.zeros(cfg.Nz)

    # Low-side PML: depth grows away from the interior, so reverse the profile
    sigma_x[: cfg.n_pml] = profile[::-1]
    sigma_z[: cfg.n_pml] = profile[::-1]
    # High-side PML: depth grows toward the outer wall, profile as-is
    sigma_x[-cfg.n_pml :] = profile
    sigma_z[-cfg.n_pml :] = profile

    return sigma_x, sigma_z


# ── Main FDTD loop ────────────────────────────────────────────────────────────

def run_fdtd(cfg, n_layout, n_eff_strip):
    """Run the 2D TE FDTD time loop and return monitored port powers.

    Returns
    -------
    dict with keys
        P_straight : time-averaged power at the through port (left wg out)
        P_cross    : time-averaged power at the cross port  (right wg out)
        P_total    : P_cross / (P_cross + P_straight), the cross fraction
    """
    Nx, Nz = cfg.Nx, cfg.Nz
    dl, dt = cfg.dl, cfg.dt

    # --- Fields ---
    Ex = np.zeros((Nx, Nz))
    Ez = np.zeros((Nx, Nz))
    Hy = np.zeros((Nx, Nz))

    # --- Permittivity ---
    eps_r = n_layout ** 2

    # --- UPML coefficients ---
    sigma_x, sigma_z = pml_coefficients(cfg, n_eff_strip)

    # Ex damped by sigma_z (1D along k)
    den_ex = 1.0 + sigma_z * dt / (2.0 * EPS0)
    Ca_ex = (1.0 - sigma_z * dt / (2.0 * EPS0)) / den_ex
    Cb_ex = (dt / EPS0) / den_ex

    # Ez damped by sigma_x (1D along i)
    den_ez = 1.0 + sigma_x * dt / (2.0 * EPS0)
    Ca_ez = (1.0 - sigma_x * dt / (2.0 * EPS0)) / den_ez
    Cb_ez = (dt / EPS0) / den_ez

    # Hy damped by combined (2D)
    SX, SZ = np.meshgrid(sigma_x, sigma_z, indexing="ij")
    sigma_hy = 0.5 * (SX + SZ)
    den_hy = 1.0 + sigma_hy * dt / (2.0 * MU0)
    Da = (1.0 - sigma_hy * dt / (2.0 * MU0)) / den_hy
    Db = (dt / MU0) / den_hy

    # --- Source: Gaussian profile around left waveguide at z = z_src ---
    x_left = cfg.x_centre - int(round(cfg.D_um / 2.0 / cfg.dl_um))
    x_right = cfg.x_centre + int(round(cfg.D_um / 2.0 / cfg.dl_um))
    sigma_src = cfg.w_um / cfg.dl_um / 2.0
    i_arr = np.arange(Nx)
    src_profile = np.exp(-0.5 * ((i_arr - x_left) / sigma_src) ** 2)

    # --- Monitor windows: 2*w_um wide around each waveguide centre at z = z_det ---
    hw = int(round(cfg.w_um / cfg.dl_um))   # half-window = w_um cells -> full = 2*w_um
    sl_left = slice(x_left - hw, x_left + hw + 1)
    sl_right = slice(x_right - hw, x_right + hw + 1)

    # Running DFT at the detector plane — extracts complex Fourier
    # amplitudes at omega0 so the time-averaged Poynting can be computed
    # as P = 0.5 * Re(conj(Ex) * Hy) instead of summing aliased snapshots.
    Ex_dft = np.zeros(Nx, dtype=np.complex128)
    Hy_dft = np.zeros(Nx, dtype=np.complex128)
    n_dft = 0

    transient_skip = 5000

    # --- Time loop (vectorised) ---
    for n in range(cfg.n_steps):
        # (a) Update Hy on interior — Yee-staggered: forward differences for H,
        # paired with the E updates' backward differences for stability.
        Hy[1:-1, 1:-1] = (
            Da[1:-1, 1:-1] * Hy[1:-1, 1:-1]
            + Db[1:-1, 1:-1] * (
                (Ex[1:-1, 2:] - Ex[1:-1, 1:-1]) / dl
                - (Ez[2:, 1:-1] - Ez[1:-1, 1:-1]) / dl
            )
        )

        # (b) Update Ex on interior
        Ex[1:-1, 1:-1] = (
            Ca_ex[1:-1][np.newaxis, :] * Ex[1:-1, 1:-1]
            + (Cb_ex[1:-1][np.newaxis, :] / eps_r[1:-1, 1:-1])
            * (Hy[1:-1, 1:-1] - Hy[1:-1, :-2]) / dl
        )

        # (c) Update Ez on interior
        Ez[1:-1, 1:-1] = (
            Ca_ez[1:-1][:, np.newaxis] * Ez[1:-1, 1:-1]
            - (Cb_ez[1:-1][:, np.newaxis] / eps_r[1:-1, 1:-1])
            * (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dl
        )

        # (d) Soft CW source on Ex at the input plane
        Ex[:, cfg.z_src] += np.sin(cfg.omega0 * n * dt) * src_profile

        # (e) Running DFT at the detector plane (every step after transient)
        if n >= transient_skip:
            phasor = np.exp(-1j * cfg.omega0 * n * dt)
            Ex_dft += Ex[:, cfg.z_det] * phasor
            Hy_dft += Hy[:, cfg.z_det] * phasor
            n_dft += 1

    # Time-averaged Poynting flux from the DFT. The phasor normalisation
    # (each component is N_dft/2 times its physical amplitude) cancels in
    # the cross/total ratio, so we leave Sz in arbitrary units.
    Sz = 0.5 * np.real(np.conj(Ex_dft) * Hy_dft)
    P_straight = float(Sz[sl_left].sum())
    P_cross = float(Sz[sl_right].sum())
    denom = P_cross + P_straight
    P_total = P_cross / denom if denom != 0.0 else 0.0

    return {
        "P_straight": P_straight,
        "P_cross": P_cross,
        "P_total": P_total,
        "Ex_dft": Ex_dft,
        "Hy_dft": Hy_dft,
        "n_dft": n_dft,
        "Ex": Ex,
        "Ez": Ez,
        "Hy": Hy,
    }