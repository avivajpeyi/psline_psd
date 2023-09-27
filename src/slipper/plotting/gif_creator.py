"""Create a gif from a series of images."""
import glob
import re

import imageio.v3 as iio

try:
    from pygifsicle import optimize

    PYGIFSICLE_INSTALLED = True
except ImportError:
    PYGIFSICLE_INSTALLED = False
import os


def create_gif(image_regex, gif_path, duration=2):
    image_filepaths = sorted(glob.glob(image_regex), key=lambda x: int(re.findall(r"\d+", x)[-1]))
    if len(image_filepaths) > 3:
        images = [iio.imread(filepath) for filepath in image_filepaths]
        iio.imwrite(gif_path, images, duration=duration, loop=0)
        if os.path.isfile(gif_path) and PYGIFSICLE_INSTALLED:
            try:
                optimize(gif_path)
            except Exception as e:
                pass
