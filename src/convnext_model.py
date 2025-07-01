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

        # Extract the number of features from the classifier head
        in_features = self.backbone.classifier[2].in_features

        # Remove the original classifier head
        self.backbone.classifier = nn.Identity()

        # Define a new custom head
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
        features = self.backbone(x)  # Shape (B, in_features)
        return self.head(features)
