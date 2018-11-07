import numpy as np
import os

from jigsaw.utils.image_helper import read_image
from jigsaw.utils.permutations import read_permutations
from jigsaw.utils.pre_processing import Image_Preprocessor



class DataLoader():
    def __init__(self, preprocessor:Image_Preprocessor, image_folder_path:str, permutation_csv_path:str, random_seed=None):
        np.random.seed(random_seed)
        self.get_image_path = self.image_path_generator(image_folder_path)
        
        self.permutations = read_permutations(permutation_csv_path)
        self.number_of_premutations = self.permutations.shape[0]
        self.range_permutations = np.arange(self.number_of_premutations)
        self.preprocessor = preprocessor

        self.number_of_except = 0 
        self.false_random = np.random.randint(self.number_of_premutations, size=200)

    def image_path_generator(self, image_folder_path):
        
        image_list = os.listdir(image_folder_path)
        image_iter = iter(image_list)

        while True:
            try:
                image_name = next(image_iter)
                yield os.path.join(image_folder_path, image_name)
            except StopIteration:
                np.random.shuffle(image_list, )
                image_iter = iter(image_list)
                image_name = next(image_iter)
                self.number_of_except += 1
                yield os.path.join(image_folder_path, image_name)


    def get_batch(self, batch_size):
        batch_puzzle = np.zeros((self.preprocessor.number_of_tiles, 
                                batch_size,
                                self.preprocessor.size_of_tiles,
                                self.preprocessor.size_of_tiles,
                                3))
        batch_permutation = np.zeros(batch_size)
        for batch in range(batch_size):
            # get permutation groundtruh one encoded
            permutation_index = np.random.randint(self.number_of_premutations) 
            #permutation_index = self.false_random[batch]
            permutation = self.permutations[permutation_index]

            image_path = next(self.get_image_path)
            image = read_image(image_path)

            puzzle = self.preprocessor.create_puzzle(image, permutation)

            batch_puzzle[:, batch, :, :, :] = puzzle
            batch_permutation[batch] = permutation_index

        return(batch_puzzle, batch_permutation)

if __name__ == "__main__":
    image_processor = Image_Preprocessor()
    dataloader = DataLoader(image_processor, "../data/images/", "../data/100_permutations.csv")
    dataloader.get_batch(20)