# model.py

import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def _make_resnet18(in_channels: int = 1,
                   num_classes: int = 2,
                   pretrained: bool = True,
                   weights_path: Optional[str] = None):
    """
    Buat ResNet-18 dengan opsi memuat bobot ImageNet (pretrained).
    Jika in_channels != 3, bobot conv1 akan diadaptasi dari bobot RGB.
    Jika weights_path diberikan, file .pth/.pt akan diload (strict=False).
    """
    # Muat model dengan API weights kalau tersedia (menghindari deprecation warning)
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # fallback ke argumen lama untuk versi torchvision lebih tua
        model = models.resnet18(pretrained=pretrained)

    # Adaptasi conv1 jika jumlah channel input berbeda
    old_conv = model.conv1
    if in_channels != old_conv.in_channels:
        new_conv = nn.Conv2d(in_channels,
                             old_conv.out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=False)
        if pretrained and hasattr(old_conv, "weight"):
            with torch.no_grad():
                old_w = old_conv.weight  # shape (out_channels, in_channels_old, kH_old, kW_old)
                kH_old, kW_old = old_w.shape[2], old_w.shape[3]
                kH_new, kW_new = new_conv.weight.shape[2], new_conv.weight.shape[3]

                if in_channels == 1:
                    # rata-rata kanal RGB -> 1 kanal, lalu resize kernel jika beda ukuran
                    w_mean = old_w.mean(dim=1, keepdim=True)  # (out,1,kH_old,kW_old)
                    if (kH_old, kW_old) != (kH_new, kW_new):
                        w_resized = F.interpolate(w_mean, size=(kH_new, kW_new), mode='bilinear', align_corners=False)
                    else:
                        w_resized = w_mean
                    new_conv.weight.copy_(w_resized)
                else:
                    # ulangi / potong bobot RGB untuk jumlah channel baru, resize kernel jika perlu
                    in_old = old_w.shape[1]
                    repeat = math.ceil(in_channels / in_old)
                    w_rep = old_w.repeat(1, repeat, 1, 1)[:, :in_channels, :, :].contiguous()
                    if (kH_old, kW_old) != (kH_new, kW_new):
                        w_rep = F.interpolate(w_rep, size=(kH_new, kW_new), mode='bilinear', align_corners=False)
                    w_rep = w_rep / repeat
                    new_conv.weight.copy_(w_rep)
        model.conv1 = new_conv

    # Hilangkan maxpool awal untuk input kecil seperti 28x28
    model.maxpool = nn.Identity()

    # Sesuaikan kepala klasifikasi (binary -> 1 output logit)
    out_features = 1 if num_classes == 2 else num_classes
    model.fc = nn.Linear(model.fc.in_features, out_features)

    # Jika diberikan path bobot, muat state_dict (dukungan untuk checkpoint dengan 'state_dict' dan module.)
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        # hapus prefix 'module.' jika ada
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state, strict=False)

    return model


class ResNetClassifier(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 pretrained: bool = True,
                 weights_path: Optional[str] = None):
        super().__init__()
        self.backbone = _make_resnet18(in_channels=in_channels,
                                       num_classes=num_classes,
                                       pretrained=pretrained,
                                       weights_path=weights_path)

    def forward(self, x):
        return self.backbone(x)


# --- Bagian untuk pengujian ---
if __name__ == "__main__":
    NUM_CLASSES = 2
    IN_CHANNELS = 1

    print("--- Menguji Model 'ResNetClassifier' (pretrained ImageNet) ---")
    # contoh inisialisasi pretrained ImageNet
    model = ResNetClassifier(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=True)
    print("Arsitektur Model:")
    print(model)

    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)

    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'ResNetClassifier' berhasil.")