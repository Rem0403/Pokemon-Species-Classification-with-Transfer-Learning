from torch import nn
from torchvision import models as tvm

METHODS = ("scratch", "frozen", "finetune")
ARCHS = ("resnet18", "resnet50", "efficientnet_b0")
_WEIGHTS = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
            "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V1,
            "efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1}


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


def head(model: nn.Module) -> nn.Linear:
    if isinstance(model, nn.Sequential):  # scratch cnn
        return model[-1]
    if hasattr(model, "fc"):  # resnet
        return model.fc
    return model.classifier[-1]  # efficientnet


def build_model(method: str, num_classes: int, pretrained: bool = True,
                arch: str = "resnet18") -> nn.Module:
    """scratch = small 4 block cnn with random weights (arch gets ignored)
    frozen = imagenet backbone, only the new last layer learns
    finetune = imagenet backbone, every layer learns"""
    if method == "scratch":
        return nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                             nn.Linear(256, num_classes))
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    model = getattr(tvm, arch)(weights=_WEIGHTS[arch] if pretrained else None)
    new_head = nn.Linear(head(model).in_features, num_classes)
    if hasattr(model, "fc"):
        model.fc = new_head
    else:
        model.classifier[-1] = new_head
    if method == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        for p in new_head.parameters():
            p.requires_grad = True
    return model


def param_groups(model: nn.Module, method: str) -> list[dict]:
    """new layers get 1e-3, pretrained layers being fine-tuned get 1e-4 so they don't get wrecked"""
    if method == "scratch":
        return [{"params": list(model.parameters()), "lr": 1e-3}]
    head_params = list(head(model).parameters())
    if method == "frozen":
        return [{"params": head_params, "lr": 1e-3}]
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": 1e-4}, {"params": head_params, "lr": 1e-3}]


def set_train_mode(model: nn.Module, method: str) -> None:
    """frozen model stays in eval mode so batchnorm keeps its imagenet stats"""
    if method == "frozen":
        model.eval()
    else:
        model.train()
