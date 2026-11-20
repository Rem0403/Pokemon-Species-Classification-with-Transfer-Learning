"""The three models compared in the project."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

METHODS = ("scratch", "frozen", "finetune")


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


def build_model(method: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """scratch: small 4-block CNN, random weights.
    frozen:   ImageNet ResNet-18, only the new last layer (fc) learns.
    finetune: ImageNet ResNet-18, every layer learns.
    """
    if method == "scratch":
        return nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                             nn.Linear(256, num_classes))
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head for our classes
    if method == "frozen":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return model


def param_groups(model: nn.Module, method: str) -> list[dict]:
    """Learning rates: new layers 1e-3; pretrained layers being fine-tuned 1e-4 (move them gently)."""
    if method == "scratch":
        return [{"params": list(model.parameters()), "lr": 1e-3}]
    fc = list(model.fc.parameters())
    if method == "frozen":
        return [{"params": fc, "lr": 1e-3}]
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    return [{"params": backbone, "lr": 1e-4}, {"params": fc, "lr": 1e-3}]


def set_train_mode(model: nn.Module, method: str) -> None:
    """Frozen model stays in eval mode so BatchNorm keeps its ImageNet statistics."""
    if method == "frozen":
        model.eval()
    else:
        model.train()
