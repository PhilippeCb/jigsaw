import numpy as np
import torch

from jigsaw.utils.data_loader import DataLoader
from jigsaw.utils.model import JigsawNet
from jigsaw.utils.pre_processing import Image_Preprocessor

class Trainer():

    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model
        self.dtype = torch.FloatTensor
        self.lr = 0.002
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                factor=0.2, patience=5, verbose=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.dtype = torch.cuda.FloatTensor

    def write_loss(self, lr_list, loss_list):
        to_write = np.zeros((2, len(lr_list)))
        to_write[0] = np.array(lr_list)
        to_write[1] = np.array(loss_list)

        np.savetxt("./loss_lr.csv", 
                   to_write.T, 
                   delimiter=",",
                   fmt="% 10.8f")

    def write_model(self):
        torch.save(self.model.state_dict(), "./first_model")
    
    def train(self, number_of_epochs, batch_size=64):

        lr_list = []
        loss_list = []
        window = 10
        last_change = 0


        while self.data_loader.number_of_except < number_of_epochs:
            images, permutation = self.data_loader.get_batch(batch_size)
            images = torch.tensor(images, requires_grad=True).permute(0, 1, 4, 2, 3).type(self.dtype)
            print(permutation)
            permutation = torch.from_numpy(permutation).long()

            

            output_permutation = self.model.forward(images).type(self.dtype)

            loss = self.loss(output_permutation, permutation)
            print(loss.item())

            # zero the parameter gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # for plotting 
            lr_list.append(self.lr)
            loss_list.append(loss.item())

            print(self.data_loader.number_of_except, self.lr)

            first_step = self.data_loader.number_of_except == 0

            if self.data_loader.number_of_except % 500 == 0 and not first_step and \
               self.data_loader.number_of_except != last_change:
                self.write_model()
                self.write_loss(lr_list, loss_list)

            elif self.data_loader.number_of_except % 20 == 0 and not first_step and \
                 self.data_loader.number_of_except != last_change:
                self.write_loss(lr_list, loss_list)
                

if __name__ == "__main__":
    preprocessor = Image_Preprocessor(normalize=True, jitter_colours=True, black_and_white_proportion=0.3)
    data_loader = DataLoader(preprocessor, "../data/images/", "../data/100_permutations.csv")
    model = JigsawNet(100).float()

    #model.load_state_dict(torch.load("./first_model"))


    trainer = Trainer(data_loader, model)
    trainer.train(10000, 50)
