#
# Transforms
#

import torch
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor


def main():
    ds_tensor = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0, torch.tensor(y), value=1
            )
        ),
    )

    ds_PIL = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=None,
        target_transform=None,
    )

    print("ds_tensor:")
    print((ds_tensor[0]))
    print("ds_PIL")
    print((ds_PIL[0]))


if __name__ == "__main__":
    main()
