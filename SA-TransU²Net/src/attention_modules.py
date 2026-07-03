# src/attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# src/attention_modules.py
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的平均和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 合并两种特征
        att_map = torch.cat([avg_out, max_out], dim=1)
        # 生成注意力图
        att_map = self.sigmoid(self.conv(att_map))
        # 应用注意力（这里是关键：确保输出维度与输入一致）
        return x * att_map  # 直接与原图相乘，保持通道数不变


class RSUNetWithAttention(nn.Module):
    """带注意力机制的RSU模块包装类"""

    def __init__(self, rsu_module, use_attention=True):
        super(RSUNetWithAttention, self).__init__()
        self.rsu = rsu_module
        self.use_attention = use_attention
        if use_attention:
            # 在RSU模块后添加空间注意力
            self.attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.rsu(x)
        if self.use_attention:
            x = self.attention(x)
        return x


if __name__ == '__main__':
    sa = SpatialAttention()
    print(sa)