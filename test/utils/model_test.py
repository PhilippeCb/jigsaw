import numpy as np
import pytest
import torch

from jigsaw.utils.model import JigsawNet

# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

class TestJigsawNet():
    def test_output_shape(self):

        x = np.random.rand(9, 20, 3, 64, 64)
        x = torch.from_numpy(x).float()
        net = JigsawNet(200)
        output = net.forward(x)

        assert output.shape = (20, 200)




if __name__=="__main__":
    