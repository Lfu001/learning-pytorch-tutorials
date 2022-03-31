#
# Initializing a Tensor
#

import numpy as np
import torch

from outputtools import print_divider

if __name__ == "__main__":

    # Directly from data
    print_divider("Directly from data")
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)

    # From a NumPy array
    print_divider("From a NumPy array")
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(x_np)

    # From another tensor
    print_divider("From another tensor")
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")
    x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")

    # With random or constant values
    print_divider("With random or constant values")
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
