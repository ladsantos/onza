import numpy as np
import matplotlib.pyplot as plt
import time
from onza.Modules import transit, input, lyman_alpha

# Generate a stellar disk
grid = transit.Grid(size=1001)
grid.draw_star([500, 500], 480)
grid.draw_planet([500, 500], 10)
grid.draw_cells(100)

# import density map
dmap = np.loadtxt('../../dump/model_earthRad-2d.dat.gz')
dmap = dmap[105:1106, 105:1106]
grid.draw_cloud(dmap)

# Plot it just for fun
# grid.plot_transit(output_file='../../dump/transit_GJ436.pdf')

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
    ab.compute_profile()

    plt.plot(ab.doppler_shift, ab.flux, label=r'$\sigma_{\mathrm{vel}}$ = '
                                              r'30 km/s')

    plt.xlabel(r'Doppler shift from Ly-$\alpha$ (km s$^{-1}$)')
    plt.ylabel(r'Flux')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../dump/test_particle.pdf')


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

    ab = lyman_alpha.Absorption(grid, cube)

    # Interpolate the emission line with the wavelength array
    # f = si.interp1d(em_line[:, 0], em_line[:, 1], kind='cubic')
    # em = f(ab.wavelength)

    curr = time.time()
    ab.compute_profile(multiprocessing=True)
    print(time.time() - curr)
    abs_prof = ab.flux + 1.0 - np.max(ab.flux)

    plt.plot(ab.doppler_shift, abs_prof,
             label=r'$\sigma_{\mathrm{vel}}$ = 30 km/s')

    plt.xlabel(r'Doppler shift from Ly-$\alpha$ (km s$^{-1}$)')
    plt.ylabel(r'Flux')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('../../dump/test_GJ436.pdf')
    plt.show()


if __name__ == '__main__':
    # test_particle()
    test_map()
