import torch
import torch.nn as nn
import torchvision.models as models

class CrossAttention(nn.Module):
    """
    跨模态注意力模块（Cross-Attention）。
    输入:
        - query: 张量，形状 (batch, query_dim)
        - key:   张量，形状 (batch, key_dim)
        - value: 张量，形状 (batch, value_dim)
    这里我们假设 key_dim = value_dim = query_dim = 512（与ResNet18最后特征相同）
    也可以通过投影层调整维度。
    """
    def __init__(self, embed_dim=512, num_heads=4):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Q、K、V 的线性映射到多头空间
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 多头注意力机制
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # 可选：输出投影
        # self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        """
        :param query: (batch, query_dim)
        :param key:   (batch, key_dim)
        :param value: (batch, value_dim)
        :return:      (batch, embed_dim)
        
        注意：MultiheadAttention 的输入要求 (batch, seq_len, embed_dim) [batch_first=True]
        """
        # 1. 保证输入至少有序列维——这里每个特征可视作单 token，seq_len=1
        #    因此需要 (batch, 1, embed_dim)
        #    若想更复杂，也可在 CNN 输出展开多个 token（如flatten spatial），这里我们保持单 token。
        # Step 1: 添加序列维
        query = query.unsqueeze(1)  # (batch, 1, embed_dim)
        key = key.unsqueeze(1)      # (batch, 1, embed_dim)
        value = value.unsqueeze(1)  # (batch, 1, embed_dim)
        # Step 2: 线性映射到 Q, K, V 空间
        Q = self.q_proj(query)      # (batch, 1, embed_dim)
        K = self.k_proj(key)        # (batch, 1, embed_dim)
        V = self.v_proj(value)      # (batch, 1, embed_dim)
        # Step 3: 注意力; multi-head 会自动处理 batch/heads
        # 输出 attn_output: (batch, 1, embed_dim)
        attn_output, attn_weights = self.attn(Q, K, V)
        # Step 4: 去除多余的序列维
        attn_output = attn_output.squeeze(1)   # (batch, embed_dim)
        # 可以加一个输出投影
        # attn_output = self.out_proj(attn_output)
        return attn_output


class VaultPredictor(nn.Module):
    """
    多模态回归模型，包含 CrossAttention 融合 OCT 与 UBM 特征，并与数值分支拼接。
    """
    def __init__(self, numeric_in_features=10):
        super(VaultPredictor, self).__init__()
        
        # OCT 分支
        resnet_oct = models.resnet18(pretrained=True)
        self.oct_backbone = nn.Sequential(*list(resnet_oct.children())[:-1])  # (batch, 512, 1, 1)
        
        # UBM 分支
        resnet_ubm = models.resnet18(pretrained=True)
        self.ubm_backbone = nn.Sequential(*list(resnet_ubm.children())[:-1])  # (batch, 512, 1, 1)
        
        # 数值分支
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        # Cross-Attention 融合模块 (OCT为Query, UBM为Key/Value)
        self.cross_attn = CrossAttention(embed_dim=512, num_heads=4)

        # 回归头：512 (CrossAttn输出) + 128 (numeric) = 640
        self.regression_head = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, oct_img, ubm_img, numeric_feat):
        """
        :param oct_img:       张量，形状 (batch, 3, H, W)（如224x224）
        :param ubm_img:       张量，形状 (batch, 3, H, W)
        :param numeric_feat:  张量，形状 (batch, 10)
        :return:              预测拱高 (batch, 1)
        """
        # 1. OCT 分支前向传播
        oct_features = self.oct_backbone(oct_img)                      # (batch, 512, 1, 1)
        oct_features = oct_features.view(oct_features.size(0), -1)     # (batch, 512)

        # 2. UBM 分支前向传播
        ubm_features = self.ubm_backbone(ubm_img)                      # (batch, 512, 1, 1)
        ubm_features = ubm_features.view(ubm_features.size(0), -1)     # (batch, 512)

        # 3. Cross-Attention 融合
        #    - Query: OCT 特征 (batch, 512)
        #    - Key/Value: UBM 特征 (batch, 512)
        #    详细维度已在 CrossAttention 注释，seq_len=1。
        ca_fused = self.cross_attn(oct_features, ubm_features, ubm_features)  # (batch, 512)

        # 4. 数值分支前向传播
        numeric_features = self.numeric_branch(numeric_feat)                 # (batch, 128)

        # 5. 拼接 CrossAttention 输出和数值分支
        fused = torch.cat([ca_fused, numeric_features], dim=1)               # (batch, 640)

        # 6. 回归头
        out = self.regression_head(fused)                                    # (batch, 1)
        return out
