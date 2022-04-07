#
# Transforms
#

import torch
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor


def main():
    ds = datasets.FashionMNIST(
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

    print(type(ds.data))


if __name__ == "__main__":
    main()
