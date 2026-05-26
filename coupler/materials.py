"""Refractive-index models for Si, SiO2, and air around 1.55 um."""

from __future__ import annotations

import numpy as np


def n_si(lam_um):
    """Crystalline silicon index, Li 1980 Sellmeier (real part)."""
    lam = np.asarray(lam_um, dtype=float)
    eps = (
        11.6858
        + 0.939816 / lam ** 2
        + 0.00810461 * lam ** 2 / (lam ** 2 - 1.1071 ** 2)
    )
    return np.sqrt(eps)


def n_sio2(lam_um):
    """Thermal SiO2 index, Malitson 1965."""
    lam = np.asarray(lam_um, dtype=float)
    l2 = lam ** 2
    eps = (
        1.0
        + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
        + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
        + 0.8974794 * l2 / (l2 - 9.896161 ** 2)
    )
    return np.sqrt(eps)


def n_air(lam_um):
    """Air ~ 1 across telecom band."""
    return 1.0