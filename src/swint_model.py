import torch.nn as nn
import torchvision.models as models

class SwinTClassifier(nn.Module):
    def __init__(self, num_classes, transfer_learning=True):
        super(SwinTClassifier, self).__init__()
        if transfer_learning:
            self.swin_transformer = models.swin_b(
                weights=models.Swin_B_Weights.IMAGENET1K_V1
            )
        else:
            self.swin_transformer = models.swin_b(weights=None)

        # Modify the classifier head
        self.swin_transformer.head = nn.Sequential(
            nn.Linear(self.swin_transformer.head.in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.swin_transformer(x)