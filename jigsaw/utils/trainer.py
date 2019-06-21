import numpy as np
from tensorboardX import SummaryWriter
import torch

from jigsaw.utils.data_loader import DataLoader
from jigsaw.utils.model import JigsawNet, JigsawNetSmall
from jigsaw.utils.pre_processing import Image_Preprocessor


class Trainer():

    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model
        self.dtype = torch.FloatTensor
        self.lr = 0.002
        self.lr = 0.01
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = torch.nn.CrossEntropyLoss()
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                         factor=0.2, patience=5, verbose=True)
        self.writer = SummaryWriter('logs/')

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.dtype_input = torch.cuda.FloatTensor
            self.dtype_target = torch.cuda.LongTensor

    def write_model(self):
        torch.save(self.model.state_dict(), "./first_model")
    
    def train(self, number_of_epochs, batch_size=64):
        i = 0

        while self.data_loader.number_of_except < number_of_epochs:
            i += 1
            images, permutation = self.data_loader.get_batch(batch_size)
            images_input = torch.tensor(images/255. - 0.5, requires_grad=True).permute(0, 1, 4, 2, 3).type(self.dtype_input)
            permutation = torch.from_numpy(permutation).type(self.dtype_target)


            # zero the parameter gradients
            self.optimizer.zero_grad()
            output_permutation = self.model.forward(images_input).type(self.dtype_input)
            loss = self.loss(output_permutation, permutation)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('loss', loss.item(), self.data_loader.number_of_except)
            self.writer.add_scalar('learning rate',
                                   self.optimizer.param_groups[0]['lr'],
                                   self.data_loader.number_of_except)
            if i % 10 == 0:
                image_to_plot = np.moveaxis(images[:, 0, :, :, :], -1, 1)
                index_permutation = np.argmax(output_permutation[0].cpu().detach().numpy())
                print(output_permutation.cpu().detach().numpy())
                output_permutation_to_plot = self.data_loader.permutations[index_permutation]
                image = self.data_loader.preprocessor.create_image_np(image_to_plot, output_permutation_to_plot)
                self.writer.add_image('imresult', image, i)
                index_permutation = permutation[0]
                print(permutation.cpu().detach().numpy())
                print("####################")
                permutation_gt = self.data_loader.permutations[index_permutation]
                image = self.data_loader.preprocessor.create_image_np(image_to_plot, permutation_gt)
                self.writer.add_image('gt', image, i)
                print(loss)

            if i % 1000 == 0:
                self.lr /= 2.


if __name__ == "__main__":
    preprocessor = Image_Preprocessor(normalize=False, jitter_colours=False, size_of_crop=111, size_of_resizing=128, black_and_white_proportion=0, size_of_tiles=32)
    data_loader = DataLoader(preprocessor, "data/images/", "data/2_permutations.csv")
    model = JigsawNetSmall(2).float()
    #model = JigsawNet(100).float()

    #model.load_state_dict(torch.load("./first_model"))

    trainer = Trainer(data_loader, model)
    trainer.train(10000000, 20)






















