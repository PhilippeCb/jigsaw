import numpy as np
from PIL import Image
import torch

from jigsaw.utils.image_helper import convert_array_to_image, convert_to_numpy, read_image, save_array_to_image


class Image_Preprocessor():

    def __init__(self, size_of_resizing=256, size_of_crop=225, number_of_tiles_per_side=3, size_of_tiles=64,
                 jitter_colours=True, normalize=True, black_and_white_proportion=0.3):
        self.size_of_resizing = size_of_resizing
        self.size_of_crop = size_of_crop
        self.number_of_tiles_per_side = number_of_tiles_per_side
        self.number_of_tiles = number_of_tiles_per_side**2
        self.size_of_tiles = size_of_tiles
        self.jitter_colours = jitter_colours
        self.normalize = normalize
        self.black_and_white_proportion = black_and_white_proportion
        self.black_and_white = False

    def get_tile(self, numpy_image, left, top, size_of_tiles, random_seed=None):
        np.random.seed(random_seed)
        if self.jitter_colours and not self.black_and_white:
            tile = np.zeros((self.size_of_tiles, self.size_of_tiles, 3))
            for c in range(3):
                left_plus_offset = left + 1 + np.random.randint(4)
                top_plus_offset = top + 1 + np.random.randint(4)
                tile[:, :, c] = numpy_image[left_plus_offset: left_plus_offset + size_of_tiles,
                                             top_plus_offset: top_plus_offset + size_of_tiles, c] 
        else:
            tile = numpy_image[left: left + size_of_tiles, 
                               top: top + size_of_tiles, :] 
        return tile

    def _get_top_left_and_leeway(self, image_size):
        image_size = np.float(image_size[1])
        size_of_possible_tiles = np.int(image_size/self.number_of_tiles_per_side)
        leeway = size_of_possible_tiles - self.size_of_tiles
        # We reduce leeway to enable spatial jittering of colours
        if self.jitter_colours and not self.black_and_white:
            leeway -= 4
        top_left_of_tiles = np.arange(self.number_of_tiles_per_side)*size_of_possible_tiles 

        return top_left_of_tiles, leeway

    def create_puzzle(self, image, permutation=None, random_seed=None):
        """
        For simplicity this code assumes the image is squared.
        """
        np.random.seed(random_seed)
        epsilon = 10**(-10)
        puzzle = np.zeros((self.number_of_tiles, self.size_of_tiles, self.size_of_tiles, 3))

        # preprocess iamges
        image = self.resize_keep_aspect_ratio(image)
        image = self.square_crop_image(image)
        image = self.convert_black_and_white(image)

        # easier for region selection
        numpy_image = convert_to_numpy(image)
        top_left_of_tiles, leeway = self._get_top_left_and_leeway(image.size)
        for i, left in enumerate(top_left_of_tiles):
            for j, top in enumerate(top_left_of_tiles):
                if leeway > 0:
                    left_plus_offset = left + np.random.randint(leeway)
                    top_plus_offset = top + np.random.randint(leeway)
                else:
                    left_plus_offset = left
                    top_plus_offset = top
                tile = self.get_tile(numpy_image, left_plus_offset, top_plus_offset, self.size_of_tiles, random_seed)
                if self.normalize:
                    tile = (tile - np.mean(tile))/(np.std(tile) + epsilon) + 1
                puzzle[i*self.number_of_tiles_per_side+j] = tile

        if permutation is not None:
            puzzle = puzzle[permutation, :, :, :]

        return puzzle

    def create_image_np(self, tiles_np, permutation):
        """
        Creates an image from a set of tiles and a permutation
        """
        num_tiles = 3
        tile_size = np.int(tiles_np.shape[2])
        image_assembled = np.zeros((3, num_tiles * tile_size, num_tiles * tile_size), dtype=np.uint8)

        # easier for region selection
        for i in range(num_tiles):
            for j in range(num_tiles):
                tile_index = int(np.where(permutation == i * num_tiles + j)[0])
                image_assembled[:, i * tile_size: (i+1) * tile_size, j * tile_size: (j+1) * tile_size] = tiles_np[tile_index]

        return image_assembled

    def convert_black_and_white(self, image):
        if np.random.rand() < self.black_and_white_proportion:
            self.black_and_white = True
            return(image.convert('L'))
        else:
            self.black_and_white = False
            return image

    def resize_keep_aspect_ratio(self, image):
        width, height = image.size
        image_shape = np.array([width, height], dtype=np.float64)
        axis_to_resize = np.argmin(np.abs(image_shape - self.size_of_resizing))

        ratio = self.size_of_resizing/image_shape[axis_to_resize]
        image_shape *= ratio
        image_shape = np.round(image_shape).astype(np.int).T
        image = image.resize(image_shape, Image.BILINEAR)
        return image

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

        return croped_image

if __name__=="__main__":
    #for i in range(100):
    image_preprocessor_mine = Image_Preprocessor()
    image = read_image("../data/images/oiseau.jpg")
    image = image_preprocessor_mine.convert_black_and_white(image)
    #image_preprocessor_mine.create_puzzle(image)
    image = image_preprocessor_mine.resize_keep_aspect_ratio(image)
    image = image_preprocessor_mine.square_crop_image(image)
    #image.save("../../test/oiseau_random_crop.jpg")
    puzzle = image_preprocessor_mine.create_puzzle(image, np.array([1, 0, 2, 3, 4, 5, 6, 7, 8]))