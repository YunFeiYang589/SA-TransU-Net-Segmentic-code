import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F  # 添加这行导入


# src/train_utils.py
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        # 简单边界检测卷积核
        self.kernel = torch.FloatTensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3)

        self.register_buffer('laplacian_kernel', self.kernel)

    def forward(self, pred, target):
        # 使用已注册的缓冲区中的卷积核
        pred_boundary = F.conv2d(torch.sigmoid(pred), self.laplacian_kernel, padding=1)
        target_boundary = F.conv2d(target.float(), self.laplacian_kernel, padding=1)
        # 边界损失
        return F.binary_cross_entropy_with_logits(pred_boundary, target_boundary)


class MAEMetric:
    """计算平均绝对误差(MAE)"""

    def __init__(self):
        self.sum_mae = 0.0
        self.num = 0

    def update(self, pred, target):
        """
        pred: [B, 1, H, W] 模型输出（未经过sigmoid）
        target: [B, 1, H, W] 二值标签（0或1）
        """
        pred = torch.sigmoid(pred).flatten(1)  # [B, H*W]
        target = target.flatten(1).float()  # [B, H*W]
        mae = torch.mean(torch.abs(pred - target), dim=1)  # [B]
        self.sum_mae += mae.sum().item()
        self.num += pred.size(0)

    def compute(self):
        return self.sum_mae / self.num if self.num > 0 else 0


class F1Metric:
    """计算F1分数(maxF1)"""

    def __init__(self, beta=1.0, thresholds=None):
        self.beta = beta
        self.thresholds = thresholds if thresholds is not None else np.arange(0.1, 1.0, 0.1)
        self.max_f1 = 0.0
        self.num = 0

    def update(self, pred, target):
        """
        pred: [B, 1, H, W] 模型输出（未经过sigmoid）
        target: [B, 1, H, W] 二值标签（0或1）
        """
        pred = torch.sigmoid(pred).flatten(1).cpu().numpy()  # [B, H*W]
        target = target.flatten(1).cpu().numpy()  # [B, H*W]

        for p, t in zip(pred, target):
            f1_scores = []
            for th in self.thresholds:
                pred_binary = (p > th).astype(np.float32)
                tp = np.sum(pred_binary * t)
                fp = np.sum(pred_binary * (1 - t))
                fn = np.sum((1 - pred_binary) * t)

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall + 1e-6)
                f1_scores.append(f1)

            self.max_f1 += max(f1_scores)
            self.num += 1

    def compute(self):
        return self.max_f1 / self.num if self.num > 0 else 0


def train_one_epoch(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader,
                    device: torch.device,
                    epoch: int,
                    lr_scheduler=None,
                    print_freq: int = 50,
                    scaler: GradScaler = None,
                    use_boundary_loss: bool = True,  # 新增参数
                    boundary_weight: float = 0.5):  # 边界损失权重

    model.train()
    metric_logger = tqdm(enumerate(data_loader), total=len(data_loader))
    loss_fn = nn.BCEWithLogitsLoss()  # 主损失函数

    # 新增：边界损失
    if use_boundary_loss:
        boundary_loss_fn = BoundaryLoss().to(device)

    total_loss = 0.0
    num_batches = 0

    for i, (images, targets) in metric_logger:
        images = images.to(device)
        targets = targets.to(device)
        # print(f"Input images shape: {images.shape}")  # 应该是 [B, 3, H, W]
        # 混合精度训练
        with autocast(enabled=scaler is not None):
            # U2Net训练时返回多个侧输出（主输出+6个侧输出）
            outputs = model(images)
            loss = 0.0

            # 计算主损失
            for output in outputs:
                loss += loss_fn(output, targets)
            loss /= len(outputs)  # 平均损失

            # 新增：计算边界损失
            if use_boundary_loss:
                boundary_loss = 0.0
                for output in outputs:
                    boundary_loss += boundary_loss_fn(output, targets)
                boundary_loss /= len(outputs)
                loss += boundary_weight * boundary_loss  # 加权合并

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        # print(f"Output shape: {outputs[0].shape}")
        # 打印训练进度
        if (i + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.set_description(
                f"Epoch: {epoch}  Loss: {loss.item():.4f}  LR: {lr:.6f}"
            )

    return total_loss / num_batches, optimizer.param_groups[0]["lr"]


@torch.no_grad()
def evaluate(model: nn.Module,
             data_loader: DataLoader,
             device: torch.device):
    model.eval()
    mae_metric = MAEMetric()
    f1_metric = F1Metric()

    for images, targets in tqdm(data_loader, desc="Validation"):
        images = images.to(device)
        targets = targets.to(device)

        # 推理时只需要主输出
        outputs = model(images)
        pred = outputs[0]  # 主输出

        # 关键修改：检查维度并调整
        # print(f"pred形状: {pred.shape}, targets形状: {targets.shape}")  # 打印形状便于调试
        if pred.dim() != targets.dim():
            # 如果预测是4维，标签是3维，压缩预测的通道维度
            if pred.dim() == 4 and targets.dim() == 3:
                pred = pred.squeeze(1)  # 从(N,1,H,W)变为(N,H,W)
            # 如果标签是4维，预测是3维，添加通道维度
            elif pred.dim() == 3 and targets.dim() == 4:
                pred = pred.unsqueeze(1)  # 从(N,H,W)变为(N,1,H,W)

        # 确保尺寸一致（如果形状仍不同，进行插值）
        if pred.shape != targets.shape:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1) if pred.dim() == 3 else pred,  # 确保是4维
                size=targets.shape[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # 恢复维度

        mae_metric.update(pred, targets)
        f1_metric.update(pred, targets)

    return mae_metric, f1_metric


def get_params_groups(model: nn.Module, weight_decay: float = 1e-4):
    """将模型参数分组，用于优化器（区分权重和偏置）"""
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if "bias" not in n and p.requires_grad],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if "bias" in n and p.requires_grad],
         "weight_decay": 0.0}
    ]
    return param_groups


def create_lr_scheduler(optimizer: torch.optim.Optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    """创建学习率调度器（余弦衰减+热身）"""
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor + (1 - warmup_factor) * alpha
        else:
            alpha = (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)
            return 0.5 * (1.0 + np.cos(alpha * np.pi))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)