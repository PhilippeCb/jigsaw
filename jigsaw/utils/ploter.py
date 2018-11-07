import numpy as np
import matplotlib.pyplot as plt

from jigsaw.utils.data_loader import DataLoader
from jigsaw.utils.pre_processing import Image_Preprocessor


def read_loss_lr(path:str):
    return np.loadtxt(path, delimiter=",")



if __name__ == "__main__":

    preprocessor = Image_Preprocessor(normalize=False, jitter_colours=True, black_and_white_proportion=0)
    data_loader = DataLoader(preprocessor, "../data/images/", "../data/100_permutations.csv")
    images, permutation = data_loader.get_batch(20)


    #plt.imshow(np.array(images[1][0], dtype=np.int))
    #plt.show()


    loss_lr = read_loss_lr("./loss_lr.csv")
    lr, loss = loss_lr.T

    N=20
    smoother_loss = np.convolve(loss, np.ones((N,))/N, mode='valid')
    #plt.plot(4500*lr)
    plt.plot(loss)
    plt.show()