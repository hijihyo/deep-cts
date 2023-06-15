import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import PretrainedConfig, PreTrainedModel

VARIANT_MAPPING = {
    "default": "vit_b_16",
    "vit_b_16": "vit_b_16",
    "vit_b_32": "vit_b_32",
    "vit_l_16": "vit_l_16",
    "vit_l_32": "vit_l_32",
    "vit_h_14": "vit_h_14",
}
WEIGHTS_MAPPING = {
    "none": None,
    "default": "DEFAULT",
    "imagenet1k_v1": "IMAGENET1K_V1",
    "imagenet1k_swag_e2e_v1": "IMAGENET1K_SWAG_E2E_V1",
    "imagenet1k_swag_linear_v1": "IMAGENET1K_SWAG_LINEAR_V1",
}


class THTViTConcatConfig(PretrainedConfig):
    def __init__(
        self,
        variant: str = "default",
        weights: str = "none",
        num_classes: int = 1000,
        **kwargs,
    ):
        if variant not in VARIANT_MAPPING.keys():
            raise ValueError(f"`variant` must be {list(VARIANT_MAPPING.keys())}, got {variant}.")
        if weights not in WEIGHTS_MAPPING.keys():
            raise ValueError(f"`weights` must be {list(WEIGHTS_MAPPING.keys())}, got {weights}.")

        self.variant = VARIANT_MAPPING[variant]
        self.weights = WEIGHTS_MAPPING[weights]
        self.num_classes = num_classes
        super(THTViTConcatConfig, self).__init__(**kwargs)


class THTViTConcat(PreTrainedModel):
    config_class = THTViTConcatConfig

    def __init__(self, config):
        super(THTViTConcat, self).__init__(config)

        vit1 = getattr(models, config.variant)(weights=config.weights, progress=True)
        vit2 = getattr(models, config.variant)(weights=config.weights, progress=True)

        self.extractor1 = create_feature_extractor(vit1, return_nodes={"getitem_5": "features"})
        self.extractor2 = create_feature_extractor(vit2, return_nodes={"getitem_5": "features"})

        self.classifier = nn.Linear(vit1.heads.head.in_features * 2, config.num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs1, inputs2, labels=None):
        is_batched = inputs1.dim() == 4
        if not is_batched:
            inputs1 = inputs1.unsqueeze(dim=0)
            inputs2 = inputs2.unsqueeze(dim=0)

        feat1 = self.extractor1(inputs1)["features"]
        feat2 = self.extractor2(inputs2)["features"]
        feat = torch.cat([feat1, feat2], dim=1)
        logits = self.classifier(feat)

        if not is_batched:
            logits = logits.squeeze(dim=0)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="default")
    parser.add_argument("--weights", type=str, default="none")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = THTViTConcatConfig(args.variant, args.weights, args.num_classes)
    device = torch.device(args.device)
    model = THTViTConcat(config).to(device)
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
