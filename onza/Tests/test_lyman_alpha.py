import numpy as np
import matplotlib.pyplot as plt
from onza.Modules import transit, input, lyman_alpha
import astropy.units as u
import timeit


# Generate a stellar disk
grid = transit.Grid(size=221)
grid.draw_star([110, 110], 109)
grid.draw_planet([110, 110], 1)
grid.draw_cells(11)

# Generate positions and velocities with a Gaussian distribution
n_particles = int(1E6)
part_pos = np.random.normal(loc=110, scale=50., size=(3, n_particles))
part_vel = np.random.normal(loc=0, scale=30., size=(3, n_particles))

# Compute the density cube
cell_bin = grid.cell_bin
cell_area = grid.cell_area
vel_range = (-300, 300)
res_element = 10
vel_bin = np.arange(vel_range[0] - res_element / 2,
                    vel_range[1] + res_element * 3 / 2,
                    res_element)
cube = input.ParticleEnsemble(part_pos, part_vel, cell_bin, vel_bin, cell_area,
                              atoms_per_particle=1E22)

ab = lyman_alpha.Absorption(grid, cube, res_element)


ab.compute_abs()

plt.plot(ab.doppler_shift, ab.flux, label=r'$\sigma_{\mathrm{vel}}$ = 30 km/s')

plt.xlabel(r'Doppler shift from Ly-$\alpha$ (km s$^{-1}$)')
plt.ylabel(r'Flux')
plt.legend()
plt.tight_layout()
plt.show()
