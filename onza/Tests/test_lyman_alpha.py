import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import scipy.interpolate as si
import time
from onza.Modules import transit, input, lyman_alpha

# import density map
dmap = np.loadtxt('../../dump/model_earthRad-2d.dat.gz')

# Sun
grid_size = 2201
grid = transit.Grid(size=grid_size)
grid.draw_star([1100, 1100], 1090)
grid.draw_planet([1100, 1100], 10)
grid.draw_cells(200)

# dmap is smaller than the grid, so we have to expand it
shape = np.shape(dmap)
add_size = (grid_size - shape[0]) // 2
conc_uplow = np.ones([add_size, shape[0]], float)
conc_sides = np.ones([add_size * 2 + shape[0], add_size], float)
dmap = np.concatenate((conc_uplow, dmap, conc_uplow), axis=0)
dmap = np.concatenate((conc_sides, dmap, conc_sides), axis=1)

#dmap = dmap[105:1106, 105:1106]
grid.draw_cloud(dmap)

# Setting up other important variables
cell_bin = grid.cell_bin
cell_area = grid.cell_area
vel_range = (-300, 300)
res_element = 1.0
vel_bin = np.arange(vel_range[0] - res_element / 2,
                    vel_range[1] + res_element * 3 / 2,
                    res_element)

# Import the Ly-alpha emission data
file = '../../dump/EmissionLine_HI.d'
em_line = np.loadtxt(file)


def test_particle():

    # Generate positions and velocities with a Gaussian distribution
    n_particles = int(1E6)
    part_pos = np.random.normal(loc=110, scale=50., size=(3, n_particles))
    part_vel = np.random.normal(loc=0, scale=30., size=(3, n_particles))

    # Compute the density cube
    cube = input.ParticleEnsemble(part_pos, part_vel, cell_bin, vel_bin,
                                  cell_area, atoms_per_particle=1E26)

    ab = lyman_alpha.Absorption(grid, cube)
    ab.fast_profile()


def test_map():

    # The distribution of velocities
    scale = 30.0
    loc = 0.0
    vels = []
    for i in range(len(vel_bin) - 1):
        vels.append((vel_bin[i] + vel_bin[i + 1]) / 2)
    vels = np.array(vels)
    vel_dist = 1 / scale / np.sqrt(2 * np.pi) * \
        np.exp(-0.5 * ((vels - loc) / scale) ** 2)

    # Finally compute the cube
    cube = input.DensityMap(dmap, vel_bin, vel_dist, cell_bin, cell_area)

    # Interpolate the emission line with the wavelength array
    f = si.interp1d(em_line[:, 0], em_line[:, 1], kind='cubic')

    ab = lyman_alpha.Absorption(grid, cube)
    em = f(ab.wavelength)
    ab.flux = np.copy(em)

    ab.fast_profile()
    plt.plot(ab.doppler_shift, ab.abs_profile)
    plt.show()


if __name__ == '__main__':
    test_particle()
    #test_map()
