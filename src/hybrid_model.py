import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class HybridSwinTabular(nn.Module):
    def __init__(
        self,
        num_classes,
        tabular_input_dim,
        tabular_hidden_dims=[64, 32],
        fusion_hidden_dim=128,
    ):
        super(HybridSwinTabular, self).__init__()

        # Pretrained Swin Transformer
        weights = Swin_T_Weights.IMAGENET1K_V1
        self.swin = swin_t(weights=weights)
        self.swin_head_in_dim = self.swin.head.in_features
        self.swin.head = nn.Identity()  # Remove classification head

        # FNN for tabular input
        tabular_layers = []
        in_dim = tabular_input_dim
        for h_dim in tabular_hidden_dims:
            tabular_layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ]
            in_dim = h_dim
        self.tabular_net = nn.Sequential(*tabular_layers)

        # Final fusion layers
        fusion_input_dim = self.swin_head_in_dim + tabular_hidden_dims[-1]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, image, tabular):
        img_feat = self.swin(image)  # (B, 768)
        tab_feat = self.tabular_net(tabular)  # (B, 32)
        combined = torch.cat((img_feat, tab_feat), dim=1)
        out = self.fusion_mlp(combined)
        return out
