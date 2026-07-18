import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# --- 新增: 空间连续性正则化 ---
def compute_tv_loss(attention_masks):
    """
    计算总变差损失，强制注意力掩膜在空间上连续成块，消除离散噪点
    attention_masks: [B, K, H, W]
    """
    h_diff = torch.abs(attention_masks[:, :, 1:, :] - attention_masks[:, :, :-1, :])
    w_diff = torch.abs(attention_masks[:, :, :, 1:] - attention_masks[:, :, :, :-1])
    return (h_diff.mean() + w_diff.mean())


def compute_orthogonality_loss(attn_softmax, epsilon=1e-8):
    B, K, N = attn_softmax.size()
    attn_norm = attn_softmax / (attn_softmax.norm(dim=-1, keepdim=True) + epsilon)
    sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
    mask = torch.ones(K, K, device=attn_softmax.device) - torch.eye(K, device=attn_softmax.device)
    loss_orth = (sim_matrix * mask.unsqueeze(0)).sum(dim=(1, 2)) / (K * (K - 1))
    return loss_orth.mean()


def compute_attention_entropy_loss(attention_masks, foreground_mask, target_entropy=0.55):
    """Prevent spatial-softmax heads from collapsing into isolated points."""
    height, width = attention_masks.shape[-2:]
    valid = F.adaptive_avg_pool2d(foreground_mask, (height, width)) > 0.10
    valid_count = valid.flatten(1).sum(dim=1).clamp(min=2).float()

    probs = attention_masks.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=(2, 3))
    normalized_entropy = entropy / valid_count.log().unsqueeze(1)
    return F.relu(target_entropy - normalized_entropy).pow(2).mean()


class CrossStageResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem_to_l3 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.layer4 = resnet.layer4
        self.downsample_l3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        f3 = self.stem_to_l3(x)
        f4 = self.layer4(f3)
        f3_down = self.downsample_l3(f3)
        return torch.cat([f3_down, f4], dim=1)  # [B, 768, 7, 7]


class PartAttentionAndGating(nn.Module):
    def __init__(self, in_channels=768, num_parts=4):
        super().__init__()
        self.num_parts = num_parts
        self.channel_dropout = nn.Dropout2d(p=0.15)
        self.attention_conv = nn.Conv2d(in_channels, num_parts, kernel_size=1)
        self.gating_fc = nn.ModuleList([nn.Linear(in_channels, 1) for _ in range(num_parts)])
        # 新增：高温软化系数，禁止单点坍缩
        self.temperature = 4.0

    def forward(self, feature_map, fg_mask):
        B, C, H, W = feature_map.size()
        feat_dropped = self.channel_dropout(feature_map)
        attn_logits = self.attention_conv(feat_dropped)

        # Temperature scaling must happen before hard masking; otherwise it weakens
        # the background penalty and allows heads to leak into empty pixels.
        attn_flat = attn_logits.view(B, self.num_parts, -1) / self.temperature
        valid_flat = (fg_mask.view(B, -1) > 0.05)
        empty_samples = valid_flat.sum(dim=1) == 0
        if empty_samples.any():
            valid_flat = valid_flat.clone()
            valid_flat[empty_samples] = True
        attn_flat = attn_flat.masked_fill(~valid_flat.unsqueeze(1), -1e4)

        attn_softmax = F.softmax(attn_flat, dim=-1)
        attention_masks = attn_softmax.view(B, self.num_parts, H, W)

        part_features = []
        gates = []
        for k in range(self.num_parts):
            mask_k = attention_masks[:, k, :, :].unsqueeze(1)
            F_k = (feature_map * mask_k).sum(dim=(2, 3))
            g_k = torch.sigmoid(self.gating_fc[k](F_k))
            part_features.append(F_k * g_k)
            gates.append(g_k)

        return torch.cat(part_features, dim=1), attention_masks, gates, attn_softmax


class TemperamentOmicsNet(nn.Module):
    def __init__(self, feature_dim=768, num_parts=4):
        super().__init__()
        self.backbone = CrossStageResNet()
        self.part_fusion = PartAttentionAndGating(feature_dim, num_parts)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        concat_dim = feature_dim + (feature_dim * num_parts)

        self.projector = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )

    def forward(self, x, fg_mask=None, return_masks=False, return_attention=False):
        M = self.backbone(x)
        F_global = self.global_pool(M).flatten(1)

        if fg_mask is None:
            bg_mask = (
                (x[:, 0:1, :, :] < -1.5)
                & (x[:, 1:2, :, :] < -1.5)
                & (x[:, 2:3, :, :] < -1.5)
            )
            fg_mask = (~bg_mask).float()

        pooled_mask = F.adaptive_avg_pool2d(fg_mask.float(), M.shape[-2:])
        hard_foreground = (pooled_mask > 0.10).float()
        F_parts_cat, masks, gates, attn_softmax = self.part_fusion(M, hard_foreground)

        V_plant = torch.cat([F_global, F_parts_cat], dim=1)
        z = F.normalize(self.projector(V_plant), p=2, dim=1)

        if return_masks:
            return z, masks, gates
        if return_attention:
            return z, attn_softmax, masks
        return z


class LargeMarginMultiViewLoss(nn.Module):
    """
    带有角度间隔 (Margin) 的对比先验损失，压缩同类内视差，拉大异类间界限。
    """

    def __init__(self, global_affinity_matrix, id_to_idx_map, temperature=0.1, alpha=0.1, margin=0.15):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.margin = margin
        self.register_buffer('global_affinity', global_affinity_matrix)
        self.id_to_idx = id_to_idx_map
        self.cached_masks = {}

    def forward(self, features, labels, sample_weights=None):
        batch_size = features.shape[0] // 2
        device = features.device

        if batch_size not in self.cached_masks:
            W_instance = torch.eye(batch_size, device=device).repeat(2, 2)
            logits_mask = torch.ones_like(W_instance) - torch.eye(2 * batch_size, device=device)
            self.cached_masks[batch_size] = (W_instance, logits_mask)

        W_instance, logits_mask = self.cached_masks[batch_size]
        labels_cat = torch.cat([labels, labels], dim=0)
        idx = [self.id_to_idx[l.item()] for l in labels_cat]
        W_tax = self.global_affinity[idx][:, idx]

        W_target = ((1.0 - self.alpha) * W_instance + self.alpha * W_tax) * logits_mask
        sum_W_target = W_target.sum(dim=1, keepdim=True).clamp(min=1e-9)
        Q_target = W_target / sum_W_target

        cosine_sim = torch.matmul(features, features.T)

        # 施加间隔惩罚
        margin_mask = (W_target > 0).float()
        penalized_sim = cosine_sim - (margin_mask * self.margin)

        sim_scaled = penalized_sim / self.temperature
        log_prob = F.log_softmax(sim_scaled, dim=1) * logits_mask

        loss_per_anchor = - (Q_target * log_prob).sum(dim=1)
        if sample_weights is None:
            return loss_per_anchor.mean()

        weights = torch.cat([sample_weights, sample_weights], dim=0).to(device)
        # Preserve equal class contribution while preferring reliable images
        # only within each class represented by the P x K sampler.
        normalized_weights = weights.clone()
        for label in labels_cat.unique():
            class_mask = labels_cat == label
            class_mean = weights[class_mask].mean().clamp_min(1e-8)
            normalized_weights[class_mask] = weights[class_mask] / class_mean
        normalized_weights = normalized_weights.clamp(0.35, 2.0)
        return (
            loss_per_anchor * normalized_weights
        ).sum() / normalized_weights.sum().clamp_min(1e-8)
