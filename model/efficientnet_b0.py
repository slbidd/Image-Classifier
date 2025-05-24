from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torch import nn

def build_model(num_classes=3):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

