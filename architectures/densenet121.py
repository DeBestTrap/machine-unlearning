"""ImageNet V1 DenseNet-121 with the task head used by BlockUnlearn."""
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class Model(nn.Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super().__init__()
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, nb_classes)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        nn.init.zeros_(self.model.classifier.bias)

    def forward(self, inputs):
        return self.model(inputs)
