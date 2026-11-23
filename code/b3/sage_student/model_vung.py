"""S-vùng (23/11): Student = nosam ĐÓNG BĂNG + "mắt" ResNet18 ImageNet ĐÓNG BĂNG + ĐẦU VÙNG nhỏ (học).

Vì sao (chẩn đoán scene_v1/chep_delta.py, troi_dau.py):
  - chưng cất cả ảnh (cc_vhn, du2_mat_ft): Student phải học lại cả ảnh -> TRÔI ở lớp nền (nhà, cây, trời, biển; RMS 5-10 mức)
    ăn mất lợi ở person (du2_mat_ft B0 s0: person +2,7 nhưng biển −2,9, cột −1,0 -> 10 lớp +0,08).
  - thầy vhn có DẠNG ĐÓNG: Y = Y_n + p_ng·(R_ir − Y_n) + p_vis·(R_vis − Y_n), R_ir = 0,3·Yv + 0,7·IR, R_vis = 0,9·Yv + 0,1·IR.
Nên: Y_n = nosam_kp841 (đóng băng, đúng nền của thầy), đầu vùng đoán (p_ng, p_vis) từ đặc trưng ResNet18 của Y và IR,
ghép theo ĐÚNG công thức thầy. Nơi đầu đoán p = 0 -> ảnh = nosam (không thể trôi). Học: BCE với bản đồ thầy (p_thay.npz).
Lúc chạy KHÔNG cần nhãn, không SAM, không hai_nhanh. Tham số học: chỉ "vung_dau".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Network

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
W_IR, W_VIS = 0.7, 0.1          # khớp b3/vung_hai_nhanh.py


class NetworkVung(nn.Module):
    def __init__(self, dim: int = 32, pretrained: bool = False, rong: int = 64):
        super().__init__()
        import torchvision
        self.goc = Network(dim)                       # nạp nosam_kp841, đóng băng
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.vung_s1 = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)   # 1/4, 64
        self.vung_s2 = r.layer2                                                      # 1/8, 128
        self.vung_s3 = r.layer3                                                      # 1/16, 256
        self.register_buffer("vung_mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("vung_std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        # đầu vùng (phần DUY NHẤT được học): 2 x 448 kênh đặc trưng (Y, IR) + 2 kênh ảnh thu nhỏ -> 2 bản đồ ở 1/4
        self.vung_dau = nn.Sequential(
            nn.Conv2d(2 * 448 + 2, rong, 1), nn.BatchNorm2d(rong), nn.LeakyReLU(0.01),
            nn.Conv2d(rong, rong, 3, padding=1), nn.BatchNorm2d(rong), nn.LeakyReLU(0.01),
            nn.Conv2d(rong, 2, 1))
        nn.init.zeros_(self.vung_dau[-1].weight)
        nn.init.constant_(self.vung_dau[-1].bias, -6.0)   # lúc đầu p ≈ 0,0025 -> ảnh ≈ nosam
        for k, p in self.named_parameters():
            p.requires_grad = k.startswith("vung_dau.")

    def train(self, mode: bool = True):
        super().train(mode)
        for m in (self.goc, self.vung_s1, self.vung_s2, self.vung_s3):
            m.eval()                                  # nền + mắt luôn eval (BN đóng băng)
        return self

    def _dac_trung(self, g, size):
        x = (g.expand(-1, 3, -1, -1) - self.vung_mean) / self.vung_std
        with torch.no_grad():
            f1 = self.vung_s1(x)
            f2 = self.vung_s2(f1)
            f3 = self.vung_s3(f2)
        up = lambda f: F.interpolate(f, size=size, mode="bilinear", align_corners=False)
        return [up(f1), up(f2), up(f3)]

    def logit_vung(self, vi, ir):
        """logit (p_ng, p_vis) ở 1/4 độ phân giải."""
        h, w = (vi.shape[2] + 3) // 4, (vi.shape[3] + 3) // 4
        nho = [F.interpolate(t, size=(h, w), mode="area") for t in (vi, ir)]
        return self.vung_dau(torch.cat(self._dac_trung(vi, (h, w)) + self._dac_trung(ir, (h, w)) + nho, 1))

    def forward(self, vi, ir):
        with torch.no_grad():
            y_n = self.goc(vi, ir)[0].clamp(0, 1)      # như ảnh nosam đã xuất (xuat_anh kẹp 0..1)
        lg = self.logit_vung(vi, ir)
        p = torch.sigmoid(F.interpolate(lg, size=vi.shape[2:], mode="bilinear", align_corners=False))
        p_ng, p_vis = p[:, :1], p[:, 1:]
        r_ir = (1 - W_IR) * vi + W_IR * ir
        r_vis = (1 - W_VIS) * vi + W_VIS * ir
        out = y_n + p_ng * (r_ir - y_n) + p_vis * (r_vis - y_n)
        return out, (lg, y_n)
