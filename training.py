import torch
import torch.optim as optim
from matplotlib import pyplot as plt

class MC(torch.nn.Module):

    def __init__(self):
        super(MC, self).__init__()
        self.layer_1 = torch.nn.Linear(33*3, 40*40*3)
        self.layer_2 = torch.nn.Linear(40*40*3, 80*80*3)

    def forward(self, x):
        output = self.layer_1(x)
        output = self.layer_2(output)
        return output

    def generate(self, x):
        plt.imshow(self(x).reshape(80, 80, 3).detach())
        plt.show()


def ingredients():

    model = MC()
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    return model, criterion, optimizer


def train(model, criterion, optimizer, samples):

    for sample in samples:
        input, target = sample

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(loss)
        # epoch_loss += loss


samples = [(torch.ones(33*3), torch.ones(80, 80, 3).reshape(19200)) for i in range(100)]
model, criterion, optimizer = ingredients()

model.generate(torch.ones(33*3))
train(model, criterion, optimizer, samples)

model.generate(torch.ones(33*3))