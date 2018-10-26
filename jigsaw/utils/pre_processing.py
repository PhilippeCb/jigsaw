import numpy as np
from PIL import Image

def convert_to_image(numpy_array):
    numpy_array = numpy_array.astype(np.uint8)
    pillow_image = Image.fromarray(numpy_array)
    return(pillow_image)

def convert_to_numpy(pillow_image):
    return(np.array(pillow_image))


def save_image(numpy_array, path):
    numpy_array = numpy_array.astype(np.uint8)
    pillow_image = Image.fromarray(numpy_array)
    pillow_image.save(path)

def read_image(path:str):
    image = Image.open(path)
    return(image)

def resize_keep_aspect_ratio(image, size=256):
    image_shape = np.array(image).shape
    image_shape = np.array([image_shape[0], image_shape[1]], dtype=np.float64)
    axis_to_resize = np.argmin(np.abs(image_shape - size))

    ratio = size/image_shape[axis_to_resize]
    image_shape *= ratio
    image_shape = np.round(image_shape).astype(np.int).T
    image = image.resize(image_shape, Image.BILINEAR)
    return(image)

def square_crop_array(array, square_size=225):
    assert array.shape[0] >= square_size
    assert array.shape[1] >= square_size

    leeway_x = array.shape[0] - square_size
    leeway_y = array.shape[1] - square_size

    new_corner_x = np.random.randint(leeway_x)
    new_corner_y = np.random.randint(leeway_y)

    croped_array = array[new_corner_x:new_corner_x+square_size, new_corner_y:new_corner_y+square_size]


    return(croped_array)

def jitter_colors(image):
    return(image)

if __name__=="__main__":
    image = read_image("../../test/oiseau.jpg")
    image = resize_keep_aspect_ratio(image)
    image = square_crop_array(np.array(image))
    save_image(image, "../../test/oiseau_random_crop.jpg")