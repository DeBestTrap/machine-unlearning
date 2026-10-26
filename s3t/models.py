"""Backbones matching local baselines, plus an explicit paper ViT option."""
from torch import nn


def build_backbone(config, num_classes, saved_spec=None):
    """Return (model, architecture spec). Reloading a saved spec never downloads."""
    pretrained = config.pretrained and saved_spec is None
    if config.model in ("vit_b_16", "tiny_vit"):
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        from torchvision.models.vision_transformer import VisionTransformer
        if config.model == "tiny_vit":
            model = VisionTransformer(image_size=16, patch_size=4, num_layers=4, num_heads=2,
                                      hidden_dim=32, mlp_dim=64, num_classes=num_classes)
        else:
            if config.model_name not in (None, "IMAGENET1K_V1"):
                raise ValueError("vit_b_16 currently supports IMAGENET1K_V1 weights")
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)
        spec = {"kind": "torchvision_vit", "model": config.model, "num_classes": num_classes,
                "weights": "IMAGENET1K_V1" if config.pretrained else None}
    elif config.model == "hf_vit":
        from transformers import ViTConfig, ViTForImageClassification
        name = config.model_name or "google/vit-base-patch16-224-in21k"
        if saved_spec is not None:
            model = ViTForImageClassification(ViTConfig.from_dict(saved_spec["hf_config"]))
        elif pretrained:
            model = ViTForImageClassification.from_pretrained(
                name, revision=config.revision, num_labels=num_classes, ignore_mismatched_sizes=True,
            )
        else:
            hf_config = ViTConfig.from_pretrained(name, revision=config.revision)
            hf_config.num_labels = num_classes
            model = ViTForImageClassification(hf_config)
        spec = {"kind": "hf_vit", "model": name, "num_classes": num_classes,
                "hf_config": model.config.to_dict()}
    else:
        from transformers import RobertaConfig, RobertaForSequenceClassification
        name = config.model_name or "roberta-base"
        if saved_spec is not None:
            hf_config = RobertaConfig.from_dict(saved_spec["hf_config"])
            model = RobertaForSequenceClassification(hf_config)
        elif config.model == "tiny_roberta":
            hf_config = RobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=4,
                                      num_attention_heads=2, intermediate_size=64,
                                      max_position_embeddings=64, num_labels=num_classes,
                                      hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
            model = RobertaForSequenceClassification(hf_config)
        elif pretrained:
            model = RobertaForSequenceClassification.from_pretrained(
                name, revision=config.revision, num_labels=num_classes,
            )
        else:
            hf_config = RobertaConfig.from_pretrained(name, revision=config.revision)
            hf_config.num_labels = num_classes
            model = RobertaForSequenceClassification(hf_config)
        # Match the existing repo's classifier initialization, including its dense layer.
        for module in model.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        spec = {"kind": "roberta", "model": name, "num_classes": num_classes,
                "hf_config": model.config.to_dict()}
    if saved_spec is not None and saved_spec["num_classes"] != num_classes:
        raise ValueError("Saved backbone has a different class count")
    return model, saved_spec or spec
