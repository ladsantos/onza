import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class Grid(object):
    """

    """
    def __init__(self, size=201):
        self.grid_size = size
        self.shape = (size, size)

    def draw_disk(self, center, radius):
        """

        Returns:

        """
        top_left = (center[0] - radius, center[1] - radius)
        bottom_right = (center[0] + radius, center[1] + radius)
        image = Image.new('1', self.shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline='white', fill='white')
        matrix = np.reshape(np.array(list(image.getdata())), self.shape) / \
            255
        return matrix


g = Grid(size=201)
star = g.draw_disk([100, 100], 100)
planet = g.draw_disk([50, 100], 20)
transit = star - planet
plt.imshow(transit)
plt.show()
