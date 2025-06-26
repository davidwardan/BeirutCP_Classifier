import torch.nn as nn


class TabularFNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=2):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.extend(
                [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
            )
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)
