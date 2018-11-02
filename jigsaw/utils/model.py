import numpy as np
import torch

# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

class AlexNet(torch.nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class JigsawNet(torch.nn.Module):
    def __init__(self, number_of_permutations):
        super(JigsawNet, self).__init__()
        self.conv_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 2 * 2, 512),
        )
        self.unifed_classification = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4608, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, number_of_permutations),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Input should have shape [9, batch , 3, 64, 64]
        """
        # TODO a optimiser
        x_out = list()
        for i in range(9):
            x_tmp = self.conv_backbone(x[i])
            x_tmp = x_tmp.view(x_tmp.size(0), 256 * 2 * 2)
            x_tmp = self.fc6(x_tmp)
            x_out.append(x_tmp)
        x = torch.cat(x_out, 1)
        x = self.unifed_classification(x)
        return x


if __name__=="__main__":
    x = np.random.rand(9, 5, 3, 64, 64)
    x = torch.from_numpy(x).float()
    net = JigsawNet(100)
    net.forward(x)
