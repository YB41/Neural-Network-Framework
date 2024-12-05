import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# The goal is to use the Fashion  Dataset to create a classifier.
# Due to this, the image has to be flattened.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # rename function
        self.linear_relu_stacks = nn.Sequential(
            nn.Linear(28*28, 512), # 28*28 because the image come sin 28x28 pixels
            nn.ReLU(), # most popular activation function
            nn.Linear(512, 512), # hidden layer with 512 output nodes
            nn.ReLU(),
            nn.Linear(512, 16) #16 due to number of classes in Dataset
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stacks(x) # takes care of the backpropagation in the model
        return logits


if __name__ == "__main__":
    # print the NN if file is run
    model = NeuralNetwork().to(device)
    print(model)