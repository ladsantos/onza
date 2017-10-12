import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class Grid(object):
    """

    """
    def __init__(self, size=201):
        self.grid_size = size
        self.shape = (size, size)

    def draw_transit(self, star_center, star_radius, planet_center,
                     planet_radius, star_value=1.0, planet_value=1.0):
        """

        Returns:

        """
        # Draw the star
        top_left = (star_center[0] - star_radius, star_center[1] - star_radius)
        bottom_right = (star_center[0] + star_radius, star_center[1] +
                        star_radius)
        image = Image.new('1', self.shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=1, fill=1)
        star = np.reshape(np.array(list(image.getdata())), self.shape) * \
            star_value
        norm = np.sum(star)  # Normalization factor is the total flux of star

        # Draw the planet
        top_left = (planet_center[0] - planet_radius,
                    planet_center[1] - planet_radius)
        bottom_right = (planet_center[0] + planet_radius,
                        planet_center[1] + planet_radius)
        image = Image.new('1', self.shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=1, fill=1)
        planet = np.reshape(np.array(list(image.getdata())), self.shape) * \
            planet_value

        # Returns the image of the transit normalized by the star flux
        transit = (star - planet) / norm
        return transit


if __name__ == "__main__":
    g = Grid(size=2001)
    _transit = g.draw_transit([1000, 1000], 1000, [500, 1000], 201)
    plt.imshow(_transit)
    plt.savefig('transit.pdf')
    plt.show()
