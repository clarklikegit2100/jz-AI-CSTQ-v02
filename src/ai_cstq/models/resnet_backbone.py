"""
ResNet backbone for BSGM-CellTrack.

Drop-in replacement for `SwinTransformerBackbone` with the same output contract:
`forward(x)` returns `[C2, C3, C4, C5]`, each `(B, out_channels, H/s, W/s)` for
s in {4, 8, 16, 32}, already projected to `out_channels` (= d_model).

Cell-TRACTR uses an ImageNet-pretrained ResNet (ResNet-18 for the bacterial
mother-machine data, ResNet-50 for the mammalian DeepCell data). Training the
backbone from scratch is a known cause of slow DETR convergence and object-query
collapse, so this backbone loads torchvision's ImageNet weights by default.
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torchvision


_RESNET_CHANNELS = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
    "resnet101": [256, 512, 1024, 2048],
}


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        arch: str = "resnet50",
        in_channels: int = 3,
        out_channels: int = 256,
        pretrained: bool = True,
        frozen_bn: bool = True,
        **_ignored,
    ):
        super().__init__()
        if arch not in _RESNET_CHANNELS:
            raise ValueError(f"Unsupported ResNet arch: {arch}")

        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)

        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=False,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    net.conv1.weight.copy_(old.weight.sum(1, keepdim=True))

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # stride 4
        self.layer2 = net.layer2   # stride 8
        self.layer3 = net.layer3   # stride 16
        self.layer4 = net.layer4   # stride 32

        dims = _RESNET_CHANNELS[arch]
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(d, out_channels, kernel_size=1) for d in dims]
        )
        for m in self.lateral_convs:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.zeros_(m.bias)

        if frozen_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # keep BN frozen regardless of module train state
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            print(f"[ResNetBackbone] missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[ResNetBackbone] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        feats = [c2, c3, c4, c5]
        return [conv(f) for conv, f in zip(self.lateral_convs, feats)]
