#
# Build the Neural Network
#

import torch
from torch import nn

from iotools.outputtools import print_divider


####################
# Define the Class #
####################
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear((28 * 28), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main():
    ###########################
    # Get Device for Training #
    ###########################
    print_divider("Get Device for Training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    print_divider("Define the Class")
    model = NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    ################
    # Model Layers #
    ################
    print_divider("Model Layers")
    input_image = torch.rand(3, 28, 28)
    print("sample minibatch:", input_image.size())

    # nn.Flatten
    print_divider("nn.Flatten", border="-", length=20)
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())

    # nn.Linear
    print_divider("nn.Linear", border="-", length=20)
    layer1 = nn.Linear(in_features=(28 * 28), out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())

    # nn.ReLU
    print_divider("nn.ReLU", border="-", length=20)
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")

    # nn.Sequential
    print_divider("nn.Sequential", border="-", length=20)
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3, 28, 28)
    logits = seq_modules(input_image)
    print("logits:")
    print(logits)

    # nn.Softmax
    print_divider("nn.Softmax", border="-", length=20)
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    print("predicted probabilities:")
    print(pred_probab)

    ####################
    # Model Parameters #
    ####################
    print_divider("Model Parameters")
    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


if __name__ == "__main__":
    main()
