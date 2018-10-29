import numpy as np
from PIL import Image
import torch

from jigsaw.utils.image_helper import convert_array_to_image, convert_to_numpy, read_image, save_array_to_image

class Image_Preprocessor():

    def __init__(self, size_of_resizing=256, size_of_crop=225, number_of_tiles_per_side=3, size_of_tiles=64):
        self.size_of_resizing = size_of_resizing
        self.size_of_crop = size_of_crop
        self.number_of_tiles_per_side = number_of_tiles_per_side
        self.number_of_tiles = number_of_tiles_per_side**2
        self.size_of_tiles = size_of_tiles

    def jitter_colors(self, puzzle):
        return(image)

    def create_puzzle(self, image):
        """
        For simplicity this code assumes the image is squared.
        """

        image_size = image.size[1]
        puzzle = np.zeros((self.number_of_tiles, self.size_of_tiles, self.size_of_tiles, 3))
        image_size = np.float(image_size)

        size_of_possible_tiles = np.int(image_size/self.number_of_tiles_per_side)
        leeway = size_of_possible_tiles - self.size_of_tiles
        #print(size_of_possible_tiles, self.size_of_tiles, leeway)
        top_left_of_tiles = np.arange(self.number_of_tiles_per_side)*size_of_possible_tiles 

        numpy_image = np.array(image)

        for i, left in enumerate(top_left_of_tiles):
            for j, top in enumerate(top_left_of_tiles):
                left_plus_offset = left + np.random.randint(leeway)
                top_plus_offset = top + np.random.randint(leeway)
                puzzle[i*self.number_of_tiles_per_side+j] = numpy_image[left_plus_offset: left_plus_offset + self.size_of_tiles, 
                                          top_plus_offset: top_plus_offset + self.size_of_tiles] 
        return(puzzle)


    def resize_keep_aspect_ratio(self, image):
        width, height = image.size
        image_shape = np.array([width, height], dtype=np.float64)
        axis_to_resize = np.argmin(np.abs(image_shape - self.size_of_resizing))

        ratio = self.size_of_resizing/image_shape[axis_to_resize]
        image_shape *= ratio
        image_shape = np.round(image_shape).astype(np.int).T
        image = image.resize(image_shape, Image.BILINEAR)
        return(image)

    def square_crop_image(self, image,):
        width, height = image.size
        assert width >= self.size_of_crop
        assert height >= self.size_of_crop

        leeway_width = width - self.size_of_crop
        leeway_height = height - self.size_of_crop

        left = np.random.randint(leeway_width)
        top = np.random.randint(leeway_height)
        right = left + self.size_of_crop
        bottom = top + self.size_of_crop

        croped_image = image.crop((left, top, right, bottom))

        return(croped_image)

if __name__=="__main__":
    #for i in range(100):
    image_preprocessor_mine = Image_Preprocessor()
    image = read_image("../../test/oiseau.jpg")
    #image_preprocessor_mine.create_puzzle(image)
    image = image_preprocessor_mine.resize_keep_aspect_ratio(image)
    image = image_preprocessor_mine.square_crop_image(image)
    image.save("../../test/oiseau_random_crop.jpg")
    image_preprocessor_mine.create_puzzle(image)
