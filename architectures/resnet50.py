"""ImageNet V2 ResNet-50 with the same trainable layers as the SU/BM study."""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class Model(nn.Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, nb_classes)
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name.startswith(("layer1.", "layer2.", "layer3.", "layer4.", "fc.")))

    def forward(self, inputs):
        return self.model(inputs)
