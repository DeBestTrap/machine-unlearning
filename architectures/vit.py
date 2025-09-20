import torch.nn as nn
from torch.nn import Module
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit_b16(num_classes: int = 10, pretrained: bool = True):
    """
    ViT-Base/16 with 224x224 input, classifier set to `num_classes`.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    in_feats = model.heads.head.in_features 
    model.heads.head = nn.Linear(in_feats, num_classes)
    return model

class Model(Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        self.model = build_vit_b16(nb_classes)
    
    def forward(self, x):
        return self.model(x)