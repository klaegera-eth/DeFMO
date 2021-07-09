import torch
import torch.nn as nn
from torch.autograd import grad


class GradNorm(nn.Module):
    def __init__(self, norm_parameters, num_outputs):
        super().__init__()
        self.norm_parameters = tuple(norm_parameters)
        self.weights = nn.Parameter(torch.ones(num_outputs))
        self.register_buffer("initial_loss", None)
        self.register_buffer("grad", torch.zeros_like(self.weights))
        # self.register_full_backward_hook(GradNorm.backward)
        self.weights.register_hook(self.get_grad)
        print(self.norm_parameters[0].shape, self.norm_parameters[1].shape)

    def forward(self, loss):
        if self.initial_loss is None:
            self.initial_loss = loss.detach()

        # g = grad(loss, layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        # print("WOWEEEEEEEEE", g.shape, self.initial_values)
        print("GRAAAAAAAD", self.weights)
        print("GRAAAAAAAD2", self.weights.grad)

        return loss * self.weights

    # def backward(self, grad_input, grad_output):
    #     print(f"BACKWAAAAARD {grad_input[0], grad_input[1].shape} {grad_output[0]}")
    #     print("GRAAAAAAAD3", self.weights.grad)
    #     print("GRAAAAAAAD4", self.weights)
    #     self.weights.grad = torch.tensor([0, 0, 0]).type_as(self.weights.grad)

    def get_grad(self, *_):
        return self.grad
