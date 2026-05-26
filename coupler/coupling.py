"""
Coupled-mode power model (Eq. 1) and lossless MZI transfer matrix (Eq. 2).

Liu, Khan, Bogaerts — ECIO 2023.
"""

import numpy as np


def power_coupling(kappa_per_um, L_um, kappa0=0.0):
    """Cross-port power coupling of the straight directional coupler.

    Eq. (1):   P = sin^2( kappa' * L + kappa_0 )

    Parameters
    ----------
    kappa_per_um : float or array
        Coupling per unit length  κ'  [rad / um].
    L_um : float
        Length of the straight coupling section  [um].
    kappa0 : float
        Lumped coupling from the bend sections  [rad].
        Defaults to 0 (straight section only).

    Returns
    -------
    P : float or array
        Cross-port power (0 to 1).
    """
    return np.sin(np.asarray(kappa_per_um) * L_um + kappa0) ** 2


def mzi_outputs(P, n_eff, dL, lam):
    """Bar and cross port intensities of a lossless MZI.

    Eq. (2) from the paper.  Input field is [0, 1]^T (TE, normalised).
    Two identical DCs with cross-coupling  P(lambda),  one arm longer
    by  dL  (um).

    The MZI transfer:

        K  @  diag( exp(-i 2pi n_eff dL / lam),  1 )  @  K  @  [0, 1]

    where  K = [[sqrt(1-P),   i sqrt(P)  ],
                [i sqrt(P),   sqrt(1-P)  ]]

    Parameters
    ----------
    P : float or array
        Cross-port power coupling of each DC.
    n_eff : float or array
        Effective index of the waveguide (for the arm phase).
    dL : float
        Path-length imbalance between the two arms  [um].
    lam : float or array
        Wavelength  [um].

    Returns
    -------
    (P_bar, P_cross) : tuple of float or array
        Bar-port and cross-port power.  Sum equals 1 (lossless).
    """
    lam   = np.asarray(lam,   dtype=float)
    P     = np.asarray(P,     dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)

    a   = np.sqrt(1.0 - P)          # straight-through amplitude
    b   = 1j * np.sqrt(P)           # cross amplitude  (carries the i)
    phi = np.exp(-1j * 2.0 * np.pi * n_eff * dL / lam)   # arm phase

    # Propagate [0, 1] through: first DC -> arm phases -> second DC
    # After first DC:          [b,       a     ]
    # After arm phases:        [b * phi, a     ]
    # After second DC:
    F_bar   = a * b * phi + b * a    # = a*b*(phi + 1)
    F_cross = b * b * phi + a * a    # = -P*phi + (1-P)

    return np.abs(F_bar) ** 2, np.abs(F_cross) ** 2