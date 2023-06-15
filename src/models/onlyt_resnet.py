from torch import nn
from torch.nn import functional as F
from torchvision import models
from transformers import PretrainedConfig, PreTrainedModel

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


class OnlyTResNetConfig(PretrainedConfig):
    def __init__(
        self, variant: str = "default", weights: str = "none", num_classes: int = 1000, **kwargs
    ):
        if variant not in VARIANT_MAPPING.keys():
            raise ValueError(f"`variant` must be {list(VARIANT_MAPPING.keys())}, got {variant}.")
        if weights not in WEIGHTS_MAPPING.keys():
            raise ValueError(f"`weights` must be {list(WEIGHTS_MAPPING.keys())}, got {weights}.")

        self.variant = VARIANT_MAPPING[variant]
        self.weights = WEIGHTS_MAPPING[weights]
        self.num_classes = num_classes
        super(OnlyTResNetConfig, self).__init__(**kwargs)


class OnlyTResNet(PreTrainedModel):
    config_class = OnlyTResNetConfig

    def __init__(self, config):
        super(OnlyTResNet, self).__init__(config)

        self.resnet = getattr(models, config.variant)(weights=config.weights, progress=True)
        if config.num_classes != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, config.num_classes)

    def forward(self, inputs1, inputs2, labels=None):
        is_batched = inputs1.dim() == 4
        if not is_batched:
            inputs1 = inputs1.unsqueeze(dim=0)

        logits = self.resnet(inputs1)

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

    config = OnlyTResNetConfig(args.variant, args.weights, args.num_classes)
    device = torch.device(args.device)
    model = OnlyTResNet(config).to(device)
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
