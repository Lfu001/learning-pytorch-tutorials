#
# Operations on Tensor
#

import torch

from outputtools import print_divider

if __name__ == "__main__":
    tensor = torch.rand(3, 4)

    # We move our tensor to the GPU if available
    # if torch.cuda.is_available():
    #     tensor = tensor.to("cuda")

    # Standard numpy-like indexing and slicing
    print_divider("Standard numpy-like indexing and slicing", length=60)
    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:, 1] = 0
    print(tensor)

    # Joining tensors
    print_divider("Joining tensors", length=60)
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    # Arithmetic operations
    print_divider("Arithmetic operations", length=60)
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)
    print("y1=\n", y1)
    print("y2=\n", y2)
    print("y3=\n", y3)

    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    print("z1=\n", z1)
    print("z2=\n", z2)
    print("z3=\n", z3)

    # Single-element tensors
    print_divider("Single-element tensors", length=60)
    agg = tensor.sum()
    agg_item = agg.item()
    print(tensor.sum(), agg_item, type(agg_item))

    # In-place operations
    print_divider("In-place operations", length=60)
    print(f"{tensor} \n")
    tensor.add_(5)
    print(tensor)
