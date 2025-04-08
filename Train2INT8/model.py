# model.py
import torch.nn as nn
import timm

class SimpleClassifier(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # Remove original head
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
