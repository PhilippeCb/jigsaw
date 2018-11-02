import numpy as np
from PIL import Image
import torch

from jigsaw.utils.image_helper import convert_array_to_image, convert_to_numpy, read_image, save_array_to_image
from jigsaw.utils.pre_processing import Image_Preprocessor

import pytest



@pytest.fixture
def image_preprocessor_instance_class():
    return Image_Preprocessor(black_and_white_proportion=0)

@pytest.fixture
def image_preprocessor_jitter_false():
    return Image_Preprocessor(jitter_colours=False, black_and_white_proportion=0)

@pytest.fixture
def image_preprocessor_normalize_false():
    return Image_Preprocessor(normalize=False, black_and_white_proportion=0)

@pytest.fixture
def image_preprocessor_black_and_white():
    return Image_Preprocessor(black_and_white_proportion=2)

@pytest.fixture
def image():
    return read_image("./jigsaw/data/images/oiseau.jpg")

class Test_Image_Preprocessor():

    def test_resize_keep_aspect_ratio(self, image_preprocessor_instance_class):
        x = convert_array_to_image(np.random.randint(255, size=(image_preprocessor_instance_class.size_of_resizing + 100, 
                                         image_preprocessor_instance_class.size_of_resizing + 2, 3)))
        resized_image = image_preprocessor_instance_class.resize_keep_aspect_ratio(x)

        assert x.size[0] < x.size[1]
        assert resized_image.size[0] == image_preprocessor_instance_class.size_of_resizing

    def test_square_crop_image(self, image_preprocessor_instance_class):
        x = convert_array_to_image(np.random.randint(255, size=(1000, 2000, 3)))

        cropped_image = image_preprocessor_instance_class.square_crop_image(x)

        assert x.size[0] != x.size[1]
        assert cropped_image.size[0] == cropped_image.size[1]
        assert cropped_image.size[0] == image_preprocessor_instance_class.size_of_crop

    def test_create_puzzle_permutation(self, image_preprocessor_instance_class):
        x = convert_array_to_image(np.random.randint(255, size=(image_preprocessor_instance_class.size_of_crop, 
                                                                image_preprocessor_instance_class.size_of_crop, 3)))

        puzzle_1 = image_preprocessor_instance_class.create_puzzle(x, random_seed=10)

        puzzle_2 = image_preprocessor_instance_class.create_puzzle(x, permutation=np.array([1, 0, 2, 3, 4, 5, 6, 7, 8]),
                                                                   random_seed=10)

        assert (puzzle_1[0] == puzzle_2[1]).all() 
        assert (puzzle_1[0] != puzzle_2[0]).any()

    def test_colour_jittering(self, image_preprocessor_instance_class, image_preprocessor_jitter_false, image):



        x = np.random.randint(255, size=(image_preprocessor_instance_class.size_of_crop, 
                                                                image_preprocessor_instance_class.size_of_crop, 3))

        tile_1 = image_preprocessor_jitter_false.get_tile(x, 5, 5, image_preprocessor_jitter_false.size_of_tiles,
                                                          random_seed=10)
        tile_2 = image_preprocessor_instance_class.get_tile(x, 5, 5, image_preprocessor_instance_class.size_of_tiles,
                                                            random_seed=10)

        assert (tile_1[2:, 2:, 0] == tile_2[:image_preprocessor_instance_class.size_of_tiles-2,
                                            :image_preprocessor_instance_class.size_of_tiles-2, 0]).all()
        assert (tile_1[1:, 4:, 1] == tile_2[:image_preprocessor_instance_class.size_of_tiles-1,
                                            :image_preprocessor_instance_class.size_of_tiles-4, 1]).all()
        assert (tile_1[1:, 2:, 2] == tile_2[:image_preprocessor_instance_class.size_of_tiles-1,
                                            :image_preprocessor_instance_class.size_of_tiles-2, 2]).all()
          
    def test_create_puzzle_colour_normalisation(self, image_preprocessor_instance_class, image_preprocessor_normalize_false):
        x = convert_array_to_image(np.random.randint(255, size=(image_preprocessor_instance_class.size_of_crop, 
                                                                image_preprocessor_instance_class.size_of_crop, 3)))

        puzzle_1 = image_preprocessor_instance_class.create_puzzle(x)
        puzzle_2 = image_preprocessor_normalize_false.create_puzzle(x)

        for puzzle in puzzle_2:
            assert np.mean(puzzle)>1
            assert np.abs(np.std(puzzle)) > 1


        for puzzle in puzzle_1:
            assert np.mean(puzzle) < 10 ** (-10)
            assert np.abs(np.std(puzzle) - 1) < 100 ** (-5)

    def test_convert_black_and_white(self, image_preprocessor_black_and_white):
        x = np.random.randint(255, size=(image_preprocessor_black_and_white.size_of_crop, 
                                         image_preprocessor_black_and_white.size_of_crop, 3))

        assert (x[:, :, 0] != x[:, :, 1]).any()

        y = image_preprocessor_black_and_white.convert_black_and_white(convert_array_to_image(x))
        y = convert_to_numpy(y)

        assert (y[:, :, 0] == y[:, :, 1]).all()

