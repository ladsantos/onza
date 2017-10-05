import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class Grid(object):
    """

    """
    def __init__(self, size=201):
        self.grid_size = size
        self.shape = (size, size)

    def draw_disk(self, center, radius, value=1.0):
        """

        Returns:

        """
        top_left = (center[0] - radius, center[1] - radius)
        bottom_right = (center[0] + radius, center[1] + radius)
        image = Image.new('1', self.shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=1, fill=1)
        disk = np.reshape(np.array(list(image.getdata())), self.shape) * value
        return disk


if __name__ == "__main__":
    g = Grid(size=2001)
    star = g.draw_disk([1000, 1000], 1000)
    planet = g.draw_disk([500, 1000], 201)
    transit = star - planet
    plt.imshow(transit)
    plt.show()
