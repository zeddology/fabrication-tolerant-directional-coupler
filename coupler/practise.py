from coupler.geometry import CouplerCrossSection, eps_function, grid_lines
import numpy as np
import matplotlib.pyplot as plt

cs = CouplerCrossSection(w_nm=450.0)
xg, yg = grid_lines(cs)
xc = 0.5 * (xg[:-1] + xg[1:])
yc = 0.5 * (yg[:-1] + yg[1:])
eps = eps_function(cs)(xc, yc)

plt.imshow(np.sqrt(eps).T, extent=[xc[0], xc[-1], yc[0], yc[-1]],
           origin="lower", aspect="equal", cmap="viridis")
plt.colorbar(label="n"); plt.xlabel("x [um]"); plt.ylabel("y [um]")
plt.show()