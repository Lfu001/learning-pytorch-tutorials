#
# Save and Load the Model
#

import torch
import torchvision.models as models


def main():
    ####################################
    # Saving and Loading Model Weights #
    ####################################
    model = models.vgg16(pretrained=True)
    torch.save(model.state_dict(), "model_weights.pth")

    # we do not specify pretrained=True, i.e. do not load default weights
    model = models.vgg16()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    print(model)

    #########################################
    # Saving and Loading Models with Shapes #
    #########################################
    torch.save(model, "model.pth")
    model = torch.load("model.pth")
    print(model)


if __name__ == "__main__":
    main()
