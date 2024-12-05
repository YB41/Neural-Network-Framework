import keras
import torch
import tensorflow
from keras.src.datasets import fashion_mnist
#from NN_Model import NeuralNetwork, device_finder

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict


class NeuralNetwork():
    def __init__(self):
        # initialize all vars here
        #regular data after download and transform
        self.train_ds = None
        self.test_ds = None
        # validation and train split for training
        self.x_validation = None
        self.x_train = None
        #dataloaders
        self.train_dl = None
        self.test_dl = None
        self.val_dl = None
        #device  and model
        self.device = None
        self.model = None

        # define the criterion and optimizer
        self.loss_fn = None
        self.optimizer = None

        #epochs
        self.n_epochs = 2

        pass
    # Download training and testing data
    def load_and_transform(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        self.train_ds = datasets.FashionMNIST('F_MNIST_data', download=True, train=True, transform=transform)
        self.test_ds = datasets.FashionMNIST('F_MNIST_data', download=True, train=False, transform=transform)
        #return train_ds, test_ds



    # split train set into training (80%) and validation set (20%)
    def train_val_split(self):
        train_num = len(self.train_ds)
        indices = list(range(train_num))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * train_num))
        self.x_validation, self.x_train = indices[:split], indices[split:]
        print("validation:", len(self.x_validation), "train:", len(self.x_train))
        #return x_train, x_validation


    # prepare dataloaders
    def dataloaders(self):
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.x_train)
        self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=64, sampler=train_sampler)
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.x_validation)
        self.val_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=64, sampler=validation_sampler)
        self.test_dl = torch.utils.data.DataLoader(self.test_ds, batch_size=64, shuffle=True)



    # visualize one example
    def visualize_example(self):
        image, label = next(iter(self.train_dl))
        print(image[0].shape, label.shape)
        desc = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
        print(desc[label[0].item()])
        plt.imshow(image[0].numpy().squeeze(), cmap='gray')
        #plt.show() #making sure it works in PC

    def network(self):
        model = nn.Sequential(OrderedDict([('fc1', nn.Linear(784, 128)),
                                           ('relu1', nn.ReLU()),
                                           ('drop1', nn.Dropout(0.25)),
                                           ('fc2', nn.Linear(128, 64)),
                                           ('relu2', nn.ReLU()),
                                           ('drop1', nn.Dropout(0.25)),
                                           ('output', nn.Linear(64, 10)),
                                           ('logsoftmax', nn.LogSoftmax(dim=1))]))
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)

        # define the criterion and optimizer
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.003)

        #return model, loss_fn, optimizer, device

    def train_validate(self, n_epochs=2):
        train_losses = []
        test_losses = []
        for epoch in range(n_epochs):
            # Set mode to training - Dropouts will be used here
            self.model.train()
            train_epoch_loss = 0
            for images, labels in self.train_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                # flatten the images to batch_size x 784
                images = images.view(images.shape[0], -1)
                # forward pass
                outputs = self.model(images)
                # backpropogation
                train_batch_loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                train_batch_loss.backward()
                # Weight updates
                self.optimizer.step()
                train_epoch_loss += train_batch_loss.item()
            else:
                # One epoch of training complete
                # calculate average training epoch loss
                train_epoch_loss = train_epoch_loss / len(self.train_dl)

                # Now Validate on testset
                with torch.no_grad():
                    test_epoch_acc = 0
                    test_epoch_loss = 0
                    # Set mode to eval - Dropouts will NOT be used here
                    self.model.eval()
                    for images, labels in self.test_dl:
                        images, labels = images.to(self.device), labels.to(self.device)
                        # flatten images to batch_size x 784
                        images = images.view(images.shape[0], -1)
                        # make predictions
                        test_outputs = self.model(images)
                        # calculate test loss
                        test_batch_loss = self.loss_fn(test_outputs, labels)
                        test_epoch_loss += test_batch_loss

                        # get probabilities, extract the class associated with highest probability
                        proba = torch.exp(test_outputs)
                        _, pred_labels = proba.topk(1, dim=1)

                        # compare actual labels and predicted labels
                        result = pred_labels == labels.view(pred_labels.shape)
                        batch_acc = torch.mean(result.type(torch.FloatTensor))
                        test_epoch_acc += batch_acc.item()
                    else:
                        # One epoch of training and validation done
                        # calculate average testing epoch loss
                        test_epoch_loss = test_epoch_loss / len(self.test_dl)
                        # calculate accuracy as correct_pred/total_samples
                        test_epoch_acc = test_epoch_acc / len(self.test_dl)
                        # save epoch losses for plotting
                        train_losses.append(train_epoch_loss)
                        test_losses.append(test_epoch_loss)
                        # print stats for this epoch
                        print(
                            f'Epoch: {epoch} -> train_loss: {train_epoch_loss:.19f}, val_loss: {test_epoch_loss:.19f}, ',
                            f'val_acc: {test_epoch_acc * 100:.2f}%')
        # Finally plot losses
        plt.plot(train_losses, label='train-loss')
        plt.plot(test_losses, label='val-loss')
        plt.legend()
        # plt.show()

    def validate(self):
        with torch.no_grad():
            batch_acc = []
            self.model.eval()
            for images, labels in self.test_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                # flatten images to batch_size x 784
                images = images.view(images.shape[0], -1)
                # make predictions and get probabilities
                proba = torch.exp(self.model(images))
                # extract the class associted with highest probability
                _, pred_labels = proba.topk(1, dim=1)
                # compare actual labels and predicted labels
                result = pred_labels == labels.view(pred_labels.shape)
                acc = torch.mean(result.type(torch.FloatTensor))
                batch_acc.append(acc.item())
            else:
                self.accuracy = torch.mean(torch.tensor(batch_acc)) * 100
                print(f'Test Accuracy: {self.accuracy:.2f}%')

    def pipeline(self):
        self.load_and_transform()
        self.train_val_split()
        self.dataloaders()
        self.visualize_example()
        self.network()
        self.train_validate()
        self.validate()
        print(f"run success with an accuracy of {self.accuracy}")
        #return self.model, self.loss_fn, self.optimizer, self.device

X = NeuralNetwork()
print(X.model)
X.pipeline()


