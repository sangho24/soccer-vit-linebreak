from __future__ import annotations


def create_resnet18(in_chans: int = 5, num_classes: int = 2, pretrained: bool = True):
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18, ResNet18_Weights
    except Exception as e:
        raise ImportError("torchvision is required for CNN model") from e

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_chans,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if pretrained:
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, : min(3, in_chans)] = old_conv.weight[:, : min(3, old_conv.weight.shape[1])]
            if in_chans > 3:
                new_conv.weight[:, 3:in_chans] = mean_w.repeat(1, in_chans - 3, 1, 1)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
