import torch.nn as nn
import torchvision.models as models
from data import datasets
from settings import device


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_classes = len(datasets["train"].classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
