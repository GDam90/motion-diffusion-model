import torch.nn as nn


class MLP_encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # create a linean layer that maps the input to the latent space
        self.motion_embedding = nn.Linear(input_dim, latent_dim)
        # self.motion_embedding = nn.Parameter(torch.randn(input_dim, latent_dim))

    def forward(self, input):
        # input needs to be flattened
        flat_input = input.reshape(input.shape[0], -1)
        output = self.motion_embedding(flat_input)
        return output