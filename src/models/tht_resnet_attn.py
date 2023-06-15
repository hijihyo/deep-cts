import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import PretrainedConfig, PreTrainedModel

from .utils import PositionalEncoding

VARIANT_MAPPING = {
    "default": "resnet18",
    "resnet18": "resnet18",
    "resnet32": "resnet32",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
}
WEIGHTS_MAPPING = {
    "none": None,
    "default": "DEFAULT",
    "imagenet1k_v1": "IMAGENET1K_V1",
}


class THTResNetAttnConfig(PretrainedConfig):
    def __init__(
        self,
        variant: str = "default",
        weights: str = "none",
        num_classes: int = 1000,
        num_attn_heads: int = 16,
        **kwargs,
    ):
        if variant not in VARIANT_MAPPING.keys():
            raise ValueError(f"`variant` must be {list(VARIANT_MAPPING.keys())}, got {variant}.")
        if weights not in WEIGHTS_MAPPING.keys():
            raise ValueError(f"`weights` must be {list(WEIGHTS_MAPPING.keys())}, got {weights}.")

        self.variant = VARIANT_MAPPING[variant]
        self.weights = WEIGHTS_MAPPING[weights]
        self.num_classes = num_classes
        self.num_attn_heads = num_attn_heads
        super(THTResNetAttnConfig, self).__init__(**kwargs)


class THTResNetAttn(PreTrainedModel):
    config_class = THTResNetAttnConfig

    def __init__(self, config):
        super(THTResNetAttn, self).__init__(config)

        resnet1 = getattr(models, config.variant)(weights=config.weights, progress=True)
        resnet2 = getattr(models, config.variant)(weights=config.weights, progress=True)

        self.extractor1 = create_feature_extractor(
            resnet1, return_nodes={"layer4.1.relu_1": "features"}
        )
        self.extractor2 = create_feature_extractor(
            resnet2, return_nodes={"layer4.1.relu_1": "features"}
        )

        self.embed_dim = resnet1.fc.in_features
        self.pos_encoder = PositionalEncoding(self.embed_dim, dropout=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.mh_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=config.num_attn_heads, dropout=0.1, batch_first=True
        )
        self.classifier = nn.Linear(self.embed_dim, config.num_classes)

    def forward(self, inputs1, inputs2, labels=None):
        is_batched = inputs1.dim() == 4
        if not is_batched:
            inputs1 = inputs1.unsqueeze(dim=0)
            inputs2 = inputs2.unsqueeze(dim=0)

        feat1 = self.extractor1(inputs1)["features"].flatten(start_dim=2).permute(0, 2, 1)
        feat2 = self.extractor2(inputs2)["features"].flatten(start_dim=2).permute(0, 2, 1)

        cls_token = self.cls_token.repeat(feat1.size(0), 1, 1)
        feat1 = torch.cat([cls_token, feat1], dim=1)
        feat1, feat2 = self.pos_encoder(feat1), self.pos_encoder(feat2)

        feat = self.mh_attn(feat1, feat2, feat2)[0][:, 0, :]
        logits = self.classifier(feat)

        if not is_batched:
            logits = logits.squeeze(dim=0)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="default")
    parser.add_argument("--weights", type=str, default="none")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = THTResNetAttnConfig(args.variant, args.weights, args.num_classes)
    device = torch.device(args.device)
    model = THTResNetAttn(config).to(device)
    print(model)

    x1 = torch.randn(16, 3, 224, 224).to(device)
    x2 = torch.randn_like(x1)
    y = torch.randint(low=0, high=args.num_classes, size=(16,)).to(device)
    out = model(x1, x2, y)
    print("Model Example")
    print("  inputs1:", x1.size())
    print("  inputs2:", x2.size())
    print("  labels:", y.size())
    print("  logits:", out["logits"].size())
    print("  loss:", out["loss"])
