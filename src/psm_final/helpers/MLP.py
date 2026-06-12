# CREDIT: ADRIEN DOERIG

from torch import nn

# nn.Module is the base class for all neural network modules in PyTorch.
# Our SimpleMLP inherits from it.
class SimpleMLP(nn.Module):

    # __init__() method in initializes the layers and attributes of the
    # neural network when an instance is created.
    # Basically, we define all the "building blocks" we need here, and we will
    # combine them in the forward(x) function.
    def __init__(self, n_input_pixels, n_hidden):
        super().__init__()
        self.flatten = nn.Flatten()              # Converts the input image of shape (channels, heigh, width) into a flattened vector of 784 elements.
        self.hidden = nn.Linear(n_input_pixels, n_hidden) # Hidden layer
        self.relu = nn.ReLU()                    # Activation function
        self.output = nn.Linear(n_hidden, 10)    # Output layer
        self.softmax = nn.Softmax(dim=1)         # Activation function

    # forward(x) defines the forward pass of the model, specifying how
    # input x flows through the layers to produce the output.
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

class MLP_3layers(nn.Module):

    # __init__() method in initializes the layers and attributes of the
    # neural network when an instance is created.
    # Basically, we define all the "building blocks" we need here, and we will
    # combine them in the forward(x) function.
    def __init__(self, n_input_pixels, n_hidden):
        super().__init__()
        self.flatten = nn.Flatten()              # Converts the input image of shape (channels, heigh, width) into a flattened vector of 784 elements.
        self.hidden1 = nn.Linear(n_input_pixels, n_hidden) # Hidden layer
        self.hidden2 = nn.Linear(n_hidden, n_hidden) # Hidden layer
        self.hidden3 = nn.Linear(n_hidden, n_hidden) # Hidden layer
        self.relu = nn.ReLU()                    # Activation function
        self.output = nn.Linear(n_hidden, 10)    # Output layer
        self.softmax = nn.Softmax(dim=1)         # Activation function

    # forward(x) defines the forward pass of the model, specifying how
    # input x flows through the layers to produce the output.
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
