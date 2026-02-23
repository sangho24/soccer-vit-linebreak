from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViTBuildConfig:
    model_name: str = "vit_base_patch16_224"
    in_chans: int = 5
    num_classes: int = 2
    pretrained: bool = True


def create_vit_model(cfg: ViTBuildConfig):
    try:
        import timm
        import torch
    except Exception as e:
        raise ImportError("timm + torch are required for ViT model") from e

    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        in_chans=cfg.in_chans,
    )

    # Some timm versions already adapt in_chans. For safety, patch embed weight expansion when needed.
    pe = getattr(model, "patch_embed", None)
    proj = getattr(pe, "proj", None)
    if proj is not None and getattr(proj, "in_channels", None) != cfg.in_chans:
        old = proj
        new_proj = torch.nn.Conv2d(
            cfg.in_chans,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            mean_w = old.weight.mean(dim=1, keepdim=True)
            c = min(old.weight.shape[1], cfg.in_chans)
            new_proj.weight[:, :c] = old.weight[:, :c]
            if cfg.in_chans > c:
                new_proj.weight[:, c:] = mean_w.repeat(1, cfg.in_chans - c, 1, 1)
            if old.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(old.bias)
        model.patch_embed.proj = new_proj

    return model


def freeze_backbone_for_linear_probe(model) -> None:
    for p in model.parameters():
        p.requires_grad = False
    head = getattr(model, "head", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True


def unfreeze_last_blocks(model, n_blocks: int = 2) -> None:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    norm = getattr(model, "norm", None)
    if norm is not None:
        for p in norm.parameters():
            p.requires_grad = True
