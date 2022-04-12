#
# Automatic Differentiation with torch.autograd
#

import torch

from iotools.outputtools import print_divider


def main():
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    ##############################################
    # Tensors, Functions and Computational graph #
    ##############################################
    print_divider("Tensors, Functions and Computational graph", length=60)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    #######################
    # Computing Gradients #
    #######################
    print_divider("Computing Gradients")
    loss.backward()
    print(w.grad)
    print(b.grad)

    ###############################
    # Disabling Gradient Tracking #
    ###############################
    print_divider("Disabling Gradient Tracking")
    z = torch.matmul(x, w) + b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.requires_grad)

    ##########################################
    # Tensor Gradients and Jacobian Products #
    ##########################################
    print_divider("Tensor Gradients and Jacobian Products", length=60)
    inp = torch.eye(5, requires_grad=True)
    out = (inp + 1).pow(2)
    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"First call\n{inp.grad}")
    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")
    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")


if __name__ == "__main__":
    main()
