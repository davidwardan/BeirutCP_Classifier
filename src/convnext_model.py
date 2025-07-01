import torch.nn as nn
import torchvision.models as models


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes, transfer_learning=True):
        super(ConvNeXtClassifier, self).__init__()
        if transfer_learning:
            self.backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.convnext_tiny(weights=None)

        # Remove the default classifier
        self.backbone.classifier = nn.Identity()

        # Global average pooling to get (B, C)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, 1, 1)

        # Get feature size (ConvNeXt-Tiny has 768 features before classifier)
        in_features = 768

        # Custom head
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone.features(x)  # (B, C, H, W)
        x = self.pool(x)               # (B, C, 1, 1)
        x = x.view(x.size(0), -1)      # Flatten to (B, C)
        return self.head(x)
