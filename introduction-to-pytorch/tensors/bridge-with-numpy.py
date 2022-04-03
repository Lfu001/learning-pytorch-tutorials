#
# Bridge with NumPy
#

import numpy as np
import torch

from outputtools import print_divider

if __name__ == "__main__":
    # Tensor to NumPy array
    print_divider("Tensor to NumPy array")
    t = torch.ones(5)
    print(f"t: {t}", type(t))
    n = t.numpy()
    print(f"n: {n}", type(n))

    # A change in the tensor reflects in the NumPy array.
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

    # NumPy array to Tensor
    print_divider("NumPy array to Tensor")
    n = np.ones(5)
    print(f"n: {n}", type(n))
    t = torch.from_numpy(n)
    print(f"t: {t}", type(t))

    # Changes in the NumPy array reflects in the tensor.
    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")
