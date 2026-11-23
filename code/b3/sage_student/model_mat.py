"""S-mắt (23/11): Student SAGE + "mắt ngữ nghĩa" = ResNet18 ImageNet ĐÓNG BĂNG (conv1 .. layer3).

Lý do: lint (L_int theo lớp) rớt vì Student 0,136M không biết vùng nào là lớp nào (đo hệ số kéo về IR: person 0,41
dù đích 0,7). Chưng cất từ hai_nhanh (ảnh Teacher theo vùng) sẽ gặp đúng vấn đề đó. Cho Student đặc trưng pretrained
(không SAM) để thử xem nó có làm theo luật theo lớp được không.

Cấu trúc: giữ NGUYÊN Network gốc; ResNet18 chạy riêng trên Y (x3 kênh) và IR (x3 kênh), chuẩn hóa ImageNet,
lấy đặc trưng layer1 (1/4, 64 kênh), layer2 (1/8, 128), layer3 (1/16, 256), phóng về cỡ x3 (1/4), nối (2 x 448 = 896 kênh)
-> chiếu 1x1 -> CỘNG vào x3. Conv cuối của phần chiếu khởi tạo 0 -> lúc đầu y hệt Network gốc.
Tham số mới tiền tố "mat_" (xuat_anh.py nhận kiến trúc từ checkpoint). ResNet18 luôn ở chế độ eval (BN đóng băng).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Network

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class NetworkMat(Network):
    def __init__(self, dim: int = 32, pretrained: bool = False):
        super().__init__(dim)
        import torchvision
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.mat_s1 = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)   # 1/4, 64
        self.mat_s2 = r.layer2                                                      # 1/8, 128
        self.mat_s3 = r.layer3                                                      # 1/16, 256
        for p in list(self.mat_s1.parameters()) + list(self.mat_s2.parameters()) + list(self.mat_s3.parameters()):
            p.requires_grad = False
        self.register_buffer("mat_mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("mat_std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.mat_chieu = nn.Sequential(nn.Conv2d(2 * 448, dim, 1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.01),
                                       nn.Conv2d(dim, dim, 1))
        nn.init.zeros_(self.mat_chieu[-1].weight)
        nn.init.zeros_(self.mat_chieu[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in (self.mat_s1, self.mat_s2, self.mat_s3):
            m.eval()
        return self

    def _dac_trung(self, g, size):
        x = (g.expand(-1, 3, -1, -1) - self.mat_mean) / self.mat_std
        with torch.no_grad():
            f1 = self.mat_s1(x)
            f2 = self.mat_s2(f1)
            f3 = self.mat_s3(f2)
        up = lambda f: F.interpolate(f, size=size, mode="bilinear", align_corners=False)
        return [up(f1), up(f2), up(f3)]

    def forward(self, vi, ir):
        src = torch.cat([vi, ir], dim=1)
        x1 = self.module1(src)
        x2 = self.module2(self.downsample1(x1))
        x3 = self.module3(self.downsample2(x2))
        x3 = x3 + self.mat_chieu(torch.cat(self._dac_trung(vi, x3.shape[2:]) + self._dac_trung(ir, x3.shape[2:]), 1))
        x4 = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x4 = self.module4(self.adjust_channels4(torch.cat([x4, x2], 1)))
        x5 = F.interpolate(x4, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x5 = self.module5(self.adjust_channels5(torch.cat([x5, x1], 1)))
        out = self.final_decoder(x5)
        return out, (x1, x2, x3, x4, x5)
