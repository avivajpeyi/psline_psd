import glob
import re

import imageio.v3 as iio
from pygifsicle import optimize


def create_gif(image_regex, gif_path, duration=1):
    image_filepaths = sorted(glob.glob(image_regex))
    images = [iio.imread(filepath) for filepath in image_filepaths]
    iio.imwrite(gif_path, images, duration=duration)
    optimize(gif_path)
