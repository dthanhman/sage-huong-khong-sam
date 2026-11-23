"""S-xa (21/11): Student SAGE + nhánh ngữ cảnh rộng.

Lý do (scene_v1/phan_tich_nut_that.py): Student gốc chỉ thu nhỏ x4, vùng nhìn hẹp -> không biết "đây là người hay
biển báo" -> loss vùng (A2) học một luật chung cả ảnh: person tăng nhưng biển báo/car giảm (đánh đổi giữa lớp).
S-xa giữ NGUYÊN Network gốc, thêm `them` tầng thu nhỏ nữa (x16 khi them=2) sau module3, rồi đi lên và CỘNG vào x3:
    x3' = x3 + up(...)      (phần còn lại của Network gốc không đổi)
Mọi tham số mới có tiền tố "xa_" -> xuat_anh.py tự nhận kiến trúc từ checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Network


def _khoi(cin, dim):
    return nn.Sequential(
        nn.Conv2d(cin, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.01),
        nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.01),
        nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.01))


class NetworkXa(Network):
    def __init__(self, dim: int = 32, them: int = 2):
        super().__init__(dim)
        self.xa_xuong = nn.ModuleList(_khoi(dim, dim) for _ in range(them))
        self.xa_ghep = nn.ModuleList(nn.Sequential(nn.Conv2d(2 * dim, dim, 1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.01))
                                     for _ in range(them))
        self.xa_len = nn.ModuleList(_khoi(dim, dim) for _ in range(them))

    def _ngu_canh(self, x3):
        h, bo = x3, []
        for m in self.xa_xuong:
            bo.append(h)
            h = m(F.max_pool2d(h, 2))
        for g, m, s in zip(self.xa_ghep, self.xa_len, reversed(bo)):
            h = F.interpolate(h, size=s.shape[2:], mode="bilinear", align_corners=False)
            h = m(g(torch.cat([h, s], 1)))
        return x3 + h

    def forward(self, vi, ir):
        src = torch.cat([vi, ir], dim=1)
        x1 = self.module1(src)
        x2 = self.module2(self.downsample1(x1))
        x3 = self._ngu_canh(self.module3(self.downsample2(x2)))
        x4 = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x4 = self.module4(self.adjust_channels4(torch.cat([x4, x2], 1)))
        x5 = F.interpolate(x4, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x5 = self.module5(self.adjust_channels5(torch.cat([x5, x1], 1)))
        out = self.final_decoder(x5)
        return out, (x1, x2, x3, x4, x5)
