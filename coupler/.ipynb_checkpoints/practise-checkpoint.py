import numpy as np
import matplotlib.pyplot as plt
from .geometry import CouplerCrossSection, eps_function, grid_lines

cs = CouplerCrossSection(w_nm=450.0)
xg, yg = grid_lines(cs)
xc = 0.5 * (xg[:-1] + xg[1:])
yc = 0.5 * (yg[:-1] + yg[1:])

eps = eps_function(cs)(xc, yc)
silicon_mask = eps > 6.0          # silicon has eps ~ 12; everything else < 3

plt.imshow(silicon_mask.T, extent=[xc[0], xc[-1], yc[0], yc[-1]],
           origin="lower", aspect="equal", cmap="gray")
plt.title("Silicon regions (from eps_function, no private imports)")
plt.xlabel("x [um]"); plt.ylabel("y [um]")
plt.show()