import torch
import torchvision
import torch.nn as nn
from torchvision.models import get_model

def build_model(name, num_classes, weights):
    model = get_model(name, weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model