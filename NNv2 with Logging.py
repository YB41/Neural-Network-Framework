from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
import time

# Create the Sacred experiment
ex = Experiment(f'nn_experiment_{int(time.time())}')
ex.observers.append(MongoObserver(url='mongodb://127.0.0.1:27017', db_name='my_database'))

# Add FileStorageObserver to log experiment results in a folder
ex.observers.append(FileStorageObserver('my_experiment_logs'))


@ex.config
def config():
    # Neural network hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 2
    momentum = 0.9
    random_seed = 42
    log_interval = 10

    # Dataset configuration (using MNIST for example)
    dataset = "FashionMNIST"
    num_classes = 10


# Define the Neural Network class
class NeuralNetwork():
    def __init__(self, config):
        self.config = config
        # Initialize all vars here
        self.accuracy = None
        self.train_ds = None
        self.test_ds = None
        # Validation and train split for training
        self.x_validation = None
        self.x_train = None
        # Dataloaders
        self.train_dl = None
        self.test_dl = None
        self.val_dl = None
        # Device and model
        self.device = None
        self.model = None

        # Define the criterion and optimizer
        self.loss_fn = None
        self.optimizer = None

        # Epochs
        self.n_epochs = config['num_epochs']

    # Download training and testing data
    def load_and_transform(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        self.train_ds = datasets.FashionMNIST('F_MNIST_data', download=True, train=True, transform=transform)
        self.test_ds = datasets.FashionMNIST('F_MNIST_data', download=True, train=False, transform=transform)

    # Split train set into training (80%) and validation set (20%)
    def train_val_split(self):
        train_num = len(self.train_ds)
        indices = list(range(train_num))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * train_num))
        self.x_validation, self.x_train = indices[:split], indices[split:]
        print("validation:", len(self.x_validation), "train:", len(self.x_train))

    # Prepare dataloaders
    def dataloaders(self):
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.x_train)
        self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.config['batch_size'],
                                                    sampler=train_sampler)
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.x_validation)
        self.val_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.config['batch_size'],
                                                  sampler=validation_sampler)
        self.test_dl = torch.utils.data.DataLoader(self.test_ds, batch_size=self.config['batch_size'], shuffle=True)

    # Build the neural network
    def network(self):
        model = nn.Sequential(OrderedDict([('fc1', nn.Linear(784, 128)),
                                           ('relu1', nn.ReLU()),
                                           ('drop1', nn.Dropout(0.25)),
                                           ('fc2', nn.Linear(128, 64)),
                                           ('relu2', nn.ReLU()),
                                           ('drop2', nn.Dropout(0.25)),
                                           ('output', nn.Linear(64, 10)),
                                           ('logsoftmax', nn.LogSoftmax(dim=1))]))
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)

        # Define the criterion and optimizer
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])

    # Training and validation
    def train_validate(self):
        train_losses = []
        test_losses = []
        for epoch in range(self.n_epochs):
            # Set mode to training - Dropouts will be used here
            self.model.train()
            train_epoch_loss = 0
            for images, labels in self.train_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.shape[0], -1)
                outputs = self.model(images)
                train_batch_loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                train_batch_loss.backward()
                self.optimizer.step()
                train_epoch_loss += train_batch_loss.item()

            # One epoch of training complete
            train_epoch_loss = train_epoch_loss / len(self.train_dl)

            # Now Validate on test set
            with torch.no_grad():
                test_epoch_acc = 0
                test_epoch_loss = 0
                self.model.eval()
                for images, labels in self.test_dl:
                    images, labels = images.to(self.device), labels.to(self.device)
                    images = images.view(images.shape[0], -1)
                    test_outputs = self.model(images)
                    test_batch_loss = self.loss_fn(test_outputs, labels)
                    test_epoch_loss += test_batch_loss

                    proba = torch.exp(test_outputs)
                    _, pred_labels = proba.topk(1, dim=1)

                    result = pred_labels == labels.view(pred_labels.shape)
                    batch_acc = torch.mean(result.type(torch.FloatTensor))
                    test_epoch_acc += batch_acc.item()

                # One epoch of validation complete
                test_epoch_loss = test_epoch_loss / len(self.test_dl)
                test_epoch_acc = test_epoch_acc / len(self.test_dl)

                # Log training and validation metrics
                ex.log_scalar('train_loss', train_epoch_loss, step=epoch)
                ex.log_scalar('val_loss', test_epoch_loss.item(), step=epoch)
                ex.log_scalar('val_accuracy', test_epoch_acc * 100, step=epoch)
                print(train_epoch_loss, test_epoch_loss.item())

                print(
                    f'Epoch: {epoch} -> train_loss: {train_epoch_loss:.4f}, val_loss: {test_epoch_loss:.4f}, val_acc: {test_epoch_acc * 100:.2f}%')

            # Save epoch losses for plotting
            train_losses.append(train_epoch_loss)
            test_losses.append(test_epoch_loss)

        # Plot training and validation losses
        plt.plot(train_losses, label='train-loss')
        plt.plot(test_losses, label='val-loss')
        plt.legend()
        # plt.show()  # Uncomment to show the plot if needed

    def validate(self):
        with torch.no_grad():
            batch_acc = []
            self.model.eval()
            for images, labels in self.test_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.shape[0], -1)
                proba = torch.exp(self.model(images))
                _, pred_labels = proba.topk(1, dim=1)
                result = pred_labels == labels.view(pred_labels.shape)
                acc = torch.mean(result.type(torch.FloatTensor))
                batch_acc.append(acc.item())

            self.accuracy = torch.mean(torch.tensor(batch_acc)) * 100
            print(f'Test Accuracy: {self.accuracy:.2f}%')

    @ex.capture
    def save_model(self):
        """Saves the model and optimizer state to a file."""
        filename = f"models/{self.config['dataset']}_model_{int(time.time())}.pth"
        torch.save({
            'epoch': self.n_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")
        ex.add_artifact(filename, content_type="application/octet-stream")

    def load_model(self, filename="model.pth"):
        """Loads the model and optimizer state from a file."""
        checkpoint = torch.load(filename)

        # Load the model state_dict
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load the optimizer state_dict
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.n_epochs = checkpoint['epoch']
        print(f"Model loaded from {filename} at epoch {self.n_epochs}")
        return self.model, self.optimizer, self.n_epochs

# Main experiment function
@ex.automain
def NN_Call(_config):
    # Initialize Neural Network with config
    nn = NeuralNetwork(config=_config)

    # Prepare data, model, and training
    nn.load_and_transform()
    nn.train_val_split()
    nn.dataloaders()
    nn.network()

    # Train and validate
    nn.train_validate()

    # Final validation accuracy
    nn.validate()
    nn.save_model()

    # Log final accuracy
    ex.log_scalar('test_accuracy', nn.accuracy.item())
    print(f"Run success with an accuracy of {nn.accuracy}")


