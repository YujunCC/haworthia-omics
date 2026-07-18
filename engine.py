import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models import (
    LargeMarginMultiViewLoss,
    compute_attention_entropy_loss,
    compute_orthogonality_loss,
    compute_tv_loss,
)
from database import build_taxonomy_affinity_matrix, update_prototypes_in_db
from dataset import HaworthiaMultiViewDataset, PKBatchSampler, train_transforms, val_transforms


def get_annealed_alpha(current_epoch, alpha_end, warmup_epochs=50):
    """
    Bring taxonomy prior to full strength early, then keep the objective fixed.
    """
    progress = min(1.0, current_epoch / max(1, warmup_epochs - 1))
    ramp = 0.5 * (1 - math.cos(math.pi * progress))
    return alpha_end * (0.25 + 0.75 * ramp)


def run_training_loop(model, device, state_bus, target_epochs, alpha_target,
                      p_classes, k_instances, resume, model_path, chkpt_path,
                      max_lambda_orth=0.02, lambda_tv=0.05,
                      lambda_entropy=0.05, target_attention_entropy=0.55,
                      prototypes_per_taxon=3, quality_aware=True,
                      minimum_quality_weight=0.35,
                      quality_sampling_strength=0.5):
    """
    流形度量引擎核心训练循环
    """
    # 1. 数据流初始化
    state_bus["message"] = "正在分析分割质量并构建类群平衡采样..."
    dataset = HaworthiaMultiViewDataset(
        is_training=True,
        transform=train_transforms,
        quality_aware=quality_aware,
        minimum_quality_weight=minimum_quality_weight,
    )
    if len(dataset) < 2:
        raise ValueError("数据量不足以启动训练，请先完成数据导入。")

    sampler = PKBatchSampler(
        dataset,
        p_classes=p_classes,
        k_instances=k_instances,
        quality_sampling_strength=(quality_sampling_strength if quality_aware else 0.0),
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    affinity, id_idx = build_taxonomy_affinity_matrix()
    state_bus["quality_summary"] = dataset.quality_summary
    state_bus["message"] = "流形度量收敛中..."

    # 2. 差异化学习率配置 (Layer-wise Learning Rate)
    # 严防破坏主干网络的低级特征提取能力，新初始化的参数组赋予较高学习率
    backbone_params = list(model.backbone.parameters())
    new_params = list(model.part_fusion.parameters()) + list(model.projector.parameters())

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 2e-5},
        {'params': new_params, 'lr': 2e-4}
    ])

    # 3. 学习率调度器：余弦退火至极小值，消除晚期优化振荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=target_epochs,
        eta_min=1e-6
    )

    start_epoch = 0

    # 4. 严谨的断点恢复机制 (Checkpoint Resume)
    if resume and os.path.exists(chkpt_path):
        chkpt = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(chkpt['model_state'])
        if 'optimizer_state' in chkpt:
            optimizer.load_state_dict(chkpt['optimizer_state'])
        if 'scheduler_state' in chkpt:
            scheduler.load_state_dict(chkpt['scheduler_state'])
        start_epoch = chkpt.get('epoch', 0) + 1
        state_bus["message"] = f"已成功从 Epoch {start_epoch} 恢复训练上下文。"

    model.train()

    # 5. 渐进式约束动力学阈值设定
    warmup_epochs = 50.0
    max_margin = 0.15
    completed = True

    # ==========================================
    # 核心 Epoch 循环
    # ==========================================
    for epoch in range(start_epoch, target_epochs):
        # 捕捉前端下发的安全中断信号
        if not state_bus.get("is_training", False):
            completed = False
            break

        state_bus["current_epoch"] = epoch + 1

        # A. 动态参数退火与预热
        current_alpha = get_annealed_alpha(epoch, alpha_target, int(warmup_epochs))
        progress = min(1.0, epoch / warmup_epochs)
        warmup_factor = 0.5 * (1 - math.cos(math.pi * progress))

        current_margin = max_margin * warmup_factor
        current_lambda_orth = max_lambda_orth * warmup_factor

        # 实例化当轮的大间隔多视角损失
        criterion = LargeMarginMultiViewLoss(
            affinity, id_idx,
            alpha=current_alpha,
            margin=current_margin
        ).to(device)

        tot_loss = 0.0
        tot_metric = 0.0
        tot_orth = 0.0
        tot_tv = 0.0
        tot_entropy = 0.0
        tot_quality = 0.0
        batches = 0

        for img_v1, mask_v1, img_v2, mask_v2, labels, quality_weights in dataloader:
            if len(labels) < 2:
                continue

            imgs = torch.cat([img_v1, img_v2], dim=0).to(device)
            foreground_masks = torch.cat([mask_v1, mask_v2], dim=0).to(device)
            labels = labels.to(device)
            quality_weights = quality_weights.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            # 正向传播：请求注意力掩膜以计算正交与连续性惩罚
            features, attn_softmax, attention_masks = model(
                imgs, fg_mask=foreground_masks, return_attention=True
            )

            # B. 损失项矩阵计算
            loss_metric = criterion(features, labels, quality_weights)
            loss_orth = compute_orthogonality_loss(attn_softmax)
            loss_tv = compute_tv_loss(attention_masks)
            loss_entropy = compute_attention_entropy_loss(
                attention_masks, foreground_masks, target_attention_entropy
            )

            # 融合总损失方程
            loss = (
                loss_metric
                + (current_lambda_orth * loss_orth)
                + (lambda_tv * loss_tv)
                + (lambda_entropy * loss_entropy)
            )

            # C. 反向传播与梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            tot_loss += loss.item()
            tot_metric += loss_metric.item()
            tot_orth += loss_orth.item()
            tot_tv += loss_tv.item()
            tot_entropy += loss_entropy.item()
            tot_quality += quality_weights.mean().item()
            batches += 1

            # D. 显式清理计算图张量，防止显存泄漏
            del (
                imgs, foreground_masks, labels, features, attn_softmax,
                attention_masks, loss, loss_metric, loss_orth, loss_tv,
                loss_entropy, quality_weights
            )

        # E. 调度器步进与状态广播
        scheduler.step()
        state_bus["loss"] = tot_loss / max(1, batches)
        state_bus["metric_loss"] = tot_metric / max(1, batches)
        state_bus["orth_loss"] = tot_orth / max(1, batches)
        state_bus["tv_loss"] = tot_tv / max(1, batches)
        state_bus["entropy_loss"] = tot_entropy / max(1, batches)
        state_bus["mean_quality_weight"] = tot_quality / max(1, batches)
        state_bus["alpha"] = current_alpha
        state_bus["margin"] = current_margin
        state_bus["orth_weight"] = current_lambda_orth

        # F. 序列化持久化断点
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'alpha': current_alpha,
            'quality_aware': quality_aware,
            'minimum_quality_weight': minimum_quality_weight,
            'quality_sampling_strength': quality_sampling_strength,
        }, chkpt_path)

    if not completed:
        state_bus["message"] = "训练已安全中断，断点已保存，未覆盖最终模型和原型。"
        return False

    rebuild_prototypes(model, device, prototypes_per_taxon)
    torch.save(model.state_dict(), model_path)
    return True


def rebuild_prototypes(model, device, prototypes_per_taxon=3, state_bus=None):
    """Extract centroids and multimodal sub-prototypes without retraining."""
    model.eval()
    val_dataset = HaworthiaMultiViewDataset(is_training=False, transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_features = {}
    with torch.no_grad():
        for batch_index, (imgs, foreground_masks, labels) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            foreground_masks = foreground_masks.to(device)
            features = model(imgs, fg_mask=foreground_masks)
            for feat, label in zip(features, labels):
                l = label.item()
                if l not in class_features:
                    class_features[l] = []
                class_features[l].append(feat.cpu())
            if state_bus is not None:
                state_bus["message"] = (
                    f"正在重建多原型：{min((batch_index + 1) * 32, len(val_dataset))}"
                    f" / {len(val_dataset)} 张图像"
                )

    update_prototypes_in_db(class_features, prototypes_per_taxon)
    return len(class_features)


def diagnose_attention(model, device, sample_limit=128):
    dataset = HaworthiaMultiViewDataset(is_training=False, transform=val_transforms)
    if not dataset:
        raise ValueError("数据库中没有可诊断的分割图像。")

    sample_count = min(sample_limit, len(dataset))
    if sample_count < len(dataset):
        indices = torch.linspace(0, len(dataset) - 1, steps=sample_count).long().tolist()
        dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    foreground_mass = []
    normalized_entropy = []
    effective_cells = []
    pair_overlap = []
    gates_all = []

    model.eval()
    with torch.no_grad():
        for imgs, foreground_masks, _ in dataloader:
            imgs = imgs.to(device)
            foreground_masks = foreground_masks.to(device)
            _, attention_masks, gates = model(
                imgs, fg_mask=foreground_masks, return_masks=True
            )
            height, width = attention_masks.shape[-2:]
            valid = F.adaptive_avg_pool2d(
                foreground_masks, (height, width)
            ) > 0.10

            foreground_mass.append((attention_masks * valid).sum(dim=(2, 3)).cpu())
            probs = attention_masks.clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=(2, 3))
            valid_count = valid.flatten(1).sum(dim=1).clamp(min=2).float()
            normalized_entropy.append((entropy / valid_count.log().unsqueeze(1)).cpu())
            effective_cells.append(entropy.exp().cpu())
            gates_all.append(torch.cat(gates, dim=1).cpu())

            flat = attention_masks.flatten(2)
            flat = flat / flat.norm(dim=2, keepdim=True).clamp_min(1e-8)
            similarities = torch.bmm(flat, flat.transpose(1, 2))
            pair_overlap.append(torch.stack([
                similarities[:, i, j]
                for i in range(attention_masks.shape[1])
                for j in range(i + 1, attention_masks.shape[1])
            ], dim=1).cpu())

    mass = torch.cat(foreground_mass)
    entropy = torch.cat(normalized_entropy)
    cells = torch.cat(effective_cells)
    gates = torch.cat(gates_all)
    overlaps = torch.cat(pair_overlap)
    heads = []
    for index in range(mass.shape[1]):
        heads.append({
            "head": index + 1,
            "foreground_mass": float(mass[:, index].mean()),
            "normalized_entropy": float(entropy[:, index].mean()),
            "effective_cells": float(cells[:, index].mean()),
            "gate": float(gates[:, index].mean()),
        })
    return {
        "sample_count": sample_count,
        "attention_grid": f"{attention_masks.shape[-2]}x{attention_masks.shape[-1]}",
        "mean_head_overlap": float(overlaps.mean()),
        "heads": heads,
    }
