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


def time_to_batch(x):
    t, n = x.size()[:2]
    x = x.view(n * t, *x.size()[2:])
    return x, n


def batch_to_time(x, n):
    nt = x.size(0)
    time = int(nt / n)
    x = x.view(time, n, *x.size()[1:])
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
        )

    def forward(self, x):
        """
        Input should have shape [9, batch , 3, 64, 64]
        """
        x, n = time_to_batch(x)
        x = self.conv_backbone(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.fc6(x)
        x = batch_to_time(x, n)
        x = self.unifed_classification(x)
        return x


class JigsawNetSmall(torch.nn.Module):
    def __init__(self, number_of_permutations):
        super(JigsawNetSmall, self).__init__()
        self.conv_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc6 = torch.nn.Sequential(
            # torch.nn.Dropout(),
            torch.nn.Linear(256 * 2 * 2, 512),
        )
        self.unifed_classification = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(),
            torch.nn.Linear(4608, number_of_permutations),
        )

    def forward(self, x):
        """
        Input should have shape [9, batch , 3, 64, 64]
        """
        x, n = time_to_batch(x)
        x = self.conv_backbone(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.fc6(x)
        x = batch_to_time(x, n)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)
        x = self.unifed_classification(x)
        return x


if __name__=="__main__":
    net = JigsawNetSmall(5).cuda()
    import time
    start = time.time()
    for i in range(100):
        x = np.random.rand(9, 5, 3, 32, 32)
        x = torch.from_numpy(x).float().cuda()
        net.forward(x)
    print((time.time() - start) / 100)





























