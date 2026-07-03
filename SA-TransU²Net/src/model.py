from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from Swin_Transformer import SwinTransformerBlock, PatchEmbed, BasicLayer
from src.attention_modules import SpatialAttention  # 假设注意力模块在同目录下


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int, use_attention=False):
        super().__init__()
        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.height = height
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        # 计算并设置输出尺寸（假设每一层下采样2倍）
        self.output_hw = (in_ch // (2 ** (height - 1)), in_ch // (2 ** (height - 1)))
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

        # 新增：是否使用注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        x = x + x_in
        # 新增：应用注意力
        if self.use_attention:
            x = self.attention(x)
        return x


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        # 计算并设置输出尺寸（假设RSU4F的下采样率为16）
        self.output_hw = (in_ch // 16, in_ch // 16)
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1, use_attention=True):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])
        self.use_attention = use_attention

        encode_list = []
        side_list = []
        for i, c in enumerate(cfg["encode"]):
            # 前4层保持RSU结构
            if i < 4:
                assert len(c) == 7
                is_rsu4f = c[4]
                use_att = c[6] if len(c) > 6 else use_attention
                if is_rsu4f:
                    encode_list.append(RSU4F(*c[1:4]))
                else:
                    encode_list.append(RSU(*c[:4], use_attention=use_att))

                # 计算特征图尺寸
                h, w = c[1] // (2 ** i), c[1] // (2 ** i)
                encode_list[-1].output_hw = (h, w)
            else:
                # 获取前一层的特征图尺寸
                if i > 0 and hasattr(encode_list[i - 1], 'output_hw'):
                    feature_h, feature_w = encode_list[i - 1].output_hw
                else:
                    feature_h, feature_w = c[1] // (2 ** i), c[1] // (2 ** i)

                # 创建Swin Transformer编码器，动态调整窗口大小
                embed_dim = c[2]
                num_heads = (embed_dim // 32,)  # 确保每个头的维度为32
                window_size = min(7, feature_h, feature_w)

                swin_encoder = SwinTransformerEncoder(
                    in_ch=c[1],
                    out_ch=c[3],
                    feature_h=feature_h,
                    feature_w=feature_w,
                    embed_dim=embed_dim,
                    depths=(2,),
                    num_heads=num_heads,
                    window_size=window_size
                )
                encode_list.append(swin_encoder)
                encode_list[-1].output_hw = (feature_h, feature_w)

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        # 解码器保持不变
        decode_list = []
        for c in cfg["decode"]:
            assert len(c) == 7
            is_rsu4f = c[4]
            use_att = c[6] if len(c) > 6 else use_attention
            if is_rsu4f:
                decode_list.append(RSU4F(*c[1:4]))
            else:
                decode_list.append(RSU(*c[:4], use_attention=use_att))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    # forward方法保持不变，无需修改
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # 后续解码流程保持不变
        # collect decode outputs
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.concat([x, x2], dim=1))
            decode_outputs.insert(0, x)

        # collect side outputs
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs
        else:
            return torch.sigmoid(x)


class SwinTransformerEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, feature_h, feature_w,
                 embed_dim=256, depths=(2,), num_heads=(8,),
                 window_size=7, mlp_ratio=4.0):
        super().__init__()
        # 确保参数兼容
        assert embed_dim % num_heads[0] == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads[0]}"

        self.conv_proj = ConvBNReLU(in_ch, embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

        self.target_h, self.target_w = feature_h, feature_w
        self.output_hw = (feature_h, feature_w)

        self.patch_embed = PatchEmbed(patch_size=4, in_c=embed_dim, embed_dim=embed_dim)

        dpr = [0.1] * depths[0]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=embed_dim,
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr
            )
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        # print("Input size: {x.shape[2:]}")
        x_proj = self.conv_proj(x)
        # print("Input size: {x_proj.shape[2:]}")
        # 调整输入格式为Swin Transformer所需的[B, H*W, C]
        x_embed, H, W = self.patch_embed(x_proj)
        # print("After patch embed: H={H}, W={W}")
        # 计算有效窗口大小（不超过特征图尺寸）
        effective_window_size = min(self.layers[0].window_size, H, W)
        for layer in self.layers:
            for block in layer.blocks:
                block.window_size = effective_window_size
                block.shift_size = effective_window_size // 2

            # 计算注意力掩码
            attn_mask = layer.create_mask(x_embed, H, W)

            x_embed, H, W = layer(x_embed, H, W)

            # 确保特征图尺寸不小于窗口大小
            if H < effective_window_size or W < effective_window_size:
                # 上采样维持特征图尺寸
                x_embed = x_embed.view(B, H, W, -1).permute(0, 3, 1, 2)
                x_embed = F.interpolate(x_embed, scale_factor=2, mode='bilinear', align_corners=False)
                H, W = x_embed.shape[2], x_embed.shape[3]
                x_embed = x_embed.flatten(2).transpose(1, 2)
        # print(f"After Swin Transformer: H={H}, W={W}, effective_window={effective_window_size}")
        # 恢复为[B, C, H, W]格式并调整尺寸
        x_transformer = x_embed.view(B, H, W, -1).permute(0, 3, 1, 2)
        x_output = self.out_proj(x_transformer)
        x_output = F.interpolate(x_output, size=(self.target_h, self.target_w),
                                 mode='bilinear', align_corners=False)

        return x_output

def u2net_full(out_ch: int = 1, use_attention=True):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side, use_attention
        "encode": [[7, 3, 32, 64, False, False, True],     # En1 启用注意力
                   [6, 64, 32, 128, False, False, True],    # En2 启用注意力
                   [5, 128, 64, 256, False, False, True],   # En3 启用注意力
                   [4, 256, 128, 512, False, False, True],  # En4 启用注意力
                   [4, 512, 256, 512, True, False, False],  # En5 不启用（深层特征）
                   [4, 512, 256, 512, True, True, False]],  # En6 不启用（深层特征）
        # height, in_ch, mid_ch, out_ch, RSU4F, side, use_attention
        "decode": [[4, 1024, 256, 512, True, True, True],   # De5 启用注意力
                   [4, 1024, 128, 256, False, True, True],  # De4 启用注意力
                   [5, 512, 64, 128, False, True, True],    # De3 启用注意力
                   [6, 256, 32, 64, False, True, True],     # De2 启用注意力
                   [7, 128, 16, 64, False, True, True]]     # De1 启用注意力
    }

    return U2Net(cfg, out_ch, use_attention=use_attention)


def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net(cfg, out_ch)


def u2net_swin(out_ch: int = 1, use_attention=True):
    cfg = {
        "encode": [
            [7, 3, 32, 64, False, False, True],
            [6, 64, 32, 128, False, False, True],
            [5, 128, 64, 256, False, False, True],
            [4, 256, 128, 512, False, False, True],
            [4, 512, 256, 512, False, False, False],
            [4, 512, 256, 512, False, True, False]
        ],  # En6 使用Swin Transformer

        # 解码器配置保持不变
        "decode": [[4, 1024, 256, 512, True, True, True],  # De5
                   [4, 1024, 128, 256, False, True, True],  # De4
                   [5, 512, 64, 128, False, True, True],  # De3
                   [6, 256, 32, 64, False, True, True],  # De2
                   [7, 128, 16, 64, False, True, True]]  # De1
    }

    return U2Net(cfg, out_ch, use_attention=use_attention)

def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    u2net = u2net_swin()
    print(u2net)
    # convert_onnx(u2net, "u2net_full.onnx")

