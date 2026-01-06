import torch.nn as nn
import torch.nn.init as init
from torch.nn import Module
from transformers import RobertaConfig, RobertaForSequenceClassification


def _init_classifier_head(head: nn.Module):
    # Xavier init for new classifier layers.
    if hasattr(head, "dense") and isinstance(head.dense, nn.Linear):
        init.xavier_uniform_(head.dense.weight)
        init.zeros_(head.dense.bias)
    if hasattr(head, "out_proj") and isinstance(head.out_proj, nn.Linear):
        init.xavier_uniform_(head.out_proj.weight)
        init.zeros_(head.out_proj.bias)


def build_roberta_base(num_classes: int = 2, pretrained: bool = True, dropout_rate: float | None = None):
    """
    RoBERTa-base for sequence classification with Xavier-initialized classifier.
    """
    if pretrained:
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_classes,
        )
    else:
        config = RobertaConfig(num_labels=num_classes)
        model = RobertaForSequenceClassification(config)

    if dropout_rate is not None:
        model.config.hidden_dropout_prob = dropout_rate
        model.config.attention_probs_dropout_prob = dropout_rate

    _init_classifier_head(model.classifier)
    return model


class Model(Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        dropout_rate = kwargs.get("dropout_rate")
        self.model = build_roberta_base(nb_classes, pretrained=True, dropout_rate=dropout_rate)
        self.pad_token_id = self.model.config.pad_token_id

    def forward(self, x):
        attention_mask = (x != self.pad_token_id).long()
        return self.model(input_ids=x, attention_mask=attention_mask).logits
