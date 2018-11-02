import torch

from jigsaw.utils.data_loader import DataLoader
from jigsaw.utils.model import JigsawNet
from jigsaw.utils.pre_processing import Image_Preprocessor

class Trainer():

    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        #self.loss = torch.nn.functional.binary_cross_entropy_with_logits()
    
    def train(self, number_of_epochs, batch_size=64):
        while self.data_loader.number_of_except < number_of_epochs:
            images, permutation = self.data_loader.get_batch(batch_size)
            images = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float()
            permutation = torch.from_numpy(permutation).double()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            output_permutation = self.model.forward(images).double()

            print(permutation, output_permutation)

            loss = self.loss(permutation, output_permutation)
            print(loss.item())
            loss.backward()
            self.optimizer.step()

    def loss(self, groundtruth, prediciton):
        loss = torch.sum((groundtruth - prediciton)**2)/groundtruth.size()[0]
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(groundtruth, prediciton)
        return(loss)

if __name__ == "__main__":
    preprocessor = Image_Preprocessor(normalize=True, jitter_colours=False, black_and_white_proportion=0)
    data_loader = DataLoader(preprocessor, "../data/images/", "../data/2_permutations.csv")
    model = JigsawNet(2).float()
    trainer = Trainer(data_loader, model)
    trainer.train(1000, 5)


    torch.save(model.state_dict(), "../data/first_model")
