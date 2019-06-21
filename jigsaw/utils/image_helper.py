import numpy as np
from PIL import Image


def convert_array_to_image(numpy_array):
    numpy_array = numpy_array.astype(np.uint8)
    pillow_image = Image.fromarray(numpy_array)
    return pillow_image


def convert_to_numpy(image):
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    return image


def read_image(path:str):
    image = Image.open(path)
    return image


def save_array_to_image(numpy_array, path:str):
    numpy_array = numpy_array.astype(np.uint8)
    pillow_image = Image.fromarray(numpy_array)
    pillow_image.save(path)
