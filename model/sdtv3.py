from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

# 导入自定义模块，mem_update 是 LIF 神经元的膜电位更新与脉冲发放函数
from lib.models.spiketrack.fuc import GateModule, downsample, upsample, FrozenBatchNorm2d, FrozenBatchNorm1d, \
    make_conv_layer
from neuron.ni_lif import mem_update

# 定义特定分辨率对应的特征图块（Patch）大小
resolution_to_patches = {256: 16, 384: 24}


class SepConv_Spike(nn.Module):
    r"""
    倒置脉冲可分离卷积 (SpikeSepConv)。
    灵感来自 MobileNetV2。论文中用此模块替代了极其耗电的 RepConv 操作，以降低能耗并保持性能。
    """
    def __init__(
            self,
            dim,
            expansion_ratio=2, # 通道扩展比例
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        
        # 脉冲神经元层 1 (时间步通常为 4)
        self.spike1 = mem_update(time_step=4)
        # 逐点卷积 (Point-wise Convolution) 用于升维
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            FrozenBatchNorm2d(med_channels)
        )

        # 脉冲神经元层 2
        self.spike2 = mem_update(time_step=4)
        # 深度卷积 (Depth-wise Convolution) 用于提取空间特征
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            FrozenBatchNorm2d(med_channels)
        )
        
        # 脉冲神经元层 3
        self.spike3 = mem_update(time_step=4)
        # 逐点卷积用于降维，恢复原始通道数
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            FrozenBatchNorm2d(dim)
        )

    def forward(self, x):
        # SNN 中的张量通常具有 5 个维度：Time, Batch, Channel, Height, Width
        T, B, C, H, W = x.shape

        # 先通过脉冲神经元发放脉冲，再进入卷积
        x = self.spike1(x)
        # 卷积操作不支持 5D 张量，需要将 T 和 B 合并 (flatten)
        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike2(x)
        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike3(x)
        x = self.pwconv2(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock_spike_SepConv(nn.Module):
    """
    网络前期的局部特征提取模块。
    结合了可分离卷积 (SepConv_Spike) 和一个类似于 FFN/MLP 的双层卷积结构，并带有残差连接。
    """
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.Conv = SepConv_Spike(dim=dim)
        self.mlp_ratio = mlp_ratio

        self.spike1 = mem_update(time_step=4)
        self.conv1 = nn.Conv2d(dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(dim * mlp_ratio)
        
        self.spike2 = mem_update(time_step=4)
        self.conv2 = nn.Conv2d(dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # 可分离卷积 + 残差连接
        x = self.Conv(x) + x
        x_feat = x
        
        # 升维卷积
        x = self.spike1(x)
        x = self.bn1(self.conv1(x.flatten(0, 1))).reshape(T, B, self.mlp_ratio * C, H, W)
        
        # 降维卷积
        x = self.spike2(x)
        x = self.bn2(self.conv2(x.flatten(0, 1))).reshape(T, B, C, H, W)
        
        # FFN 部分的残差连接
        x = x_feat + x
        return x


class MS_MLP(nn.Module):
    """
    多阶脉冲多层感知机 (Multi-Spike MLP)。
    作用相当于标准 Transformer 中的 FFN（前馈神经网络）。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, frozen=True, drop=0.0, layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 使用 1D 卷积替代全连接层 (Linear) 来处理展平后的空间维度
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_spike = mem_update(time_step=4)
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_spike = mem_update(time_step=4)
        
        if frozen:
            self.fc1_bn = FrozenBatchNorm1d(hidden_features)
            self.fc2_bn = FrozenBatchNorm1d(out_features)
        else:
            self.fc1_bn = nn.BatchNorm1d(hidden_features)
            self.fc2_bn = nn.BatchNorm1d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        # 展平 H 和 W 维度，变为 (T, B, C, N)
        x = x.flatten(3)
        
        x = self.fc1_spike(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        
        x = self.fc2_spike(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        return x


class MS_Attention_linear_3d(nn.Module):
    """
    高效的脉冲驱动自注意力模块 (E-SDSA)。
    论文中提到，此模块使用线性操作代替标准卷积来生成 Q、K、V，极大地提升了效率。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, lamda_ratio=1, frozen=True, resolution=384):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio
        
        # 定义各类脉冲神经元
        self.head_spike = mem_update(time_step=4)
        self.q_spike = mem_update(time_step=4)
        self.k_spike = mem_update(time_step=4)
        self.v_spike = mem_update(time_step=4)
        self.attn_spike = mem_update(time_step=4)
        self.resolution = resolution
        norm_layer = FrozenBatchNorm2d if frozen else nn.BatchNorm2d

        # 使用轻量级操作生成 Q, K, V 和最终的投影
        self.q_conv = make_conv_layer(dim, dim, norm_layer)
        self.k_conv = make_conv_layer(dim, dim, norm_layer)
        self.v_conv = make_conv_layer(dim, int(dim * lamda_ratio), norm_layer)
        self.proj_conv = make_conv_layer(int(dim * lamda_ratio), dim, norm_layer)

        # 针对目标跟踪任务（Search 和 Template 机制）的位置编码
        if resolution not in resolution_to_patches:
            raise ValueError(f"Unsupported resolution: {resolution}")
        patches = resolution_to_patches[resolution]
        patch_size = patches * patches
        self.s_pos_embed = nn.Parameter(torch.zeros(1, dim, patch_size))
        self.t_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim, patch_size)) for _ in range(4)
        ])

    def _apply_positional_encoding(self, x, branch, T=None):
        # 为 Search（搜索区域）和 Template（模板区域）添加不同的位置编码
        T, B, C, H, W = x.shape
        N = H * W
        x = x.view(T, B, C, N)

        if branch == 'search':
            x = x + self.s_pos_embed
        elif branch == 'template':
            out_list = [x[t] + self.t_pos_embeds[min(t, 3)] for t in range(T)]
            x = torch.stack(out_list, dim=0)
        else:
            raise ValueError(f"Unsupported branch: {branch}")
        return x.view(T, B, C, H, W)

    def forward(self, x, branch):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self._apply_positional_encoding(x, branch, T)
        x = self.head_spike(x)
        x = x.view(T * B, C, H, W)

        # 生成 Q, K, V
        q = self.q_conv(x).reshape(T, B, C, N)
        k = self.k_conv(x).reshape(T, B, C, N)
        v = self.v_conv(x).reshape(T, B, C_v, N)

        def reshape_for_attention(tensor, num_channels):
            # 将张量重塑为多头注意力的形状: (T, B, Head, N, C/Head)
            return (tensor.transpose(-1, -2)
                    .reshape(T, B, N, self.num_heads, num_channels // self.num_heads)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous())

        # Q, K, V 在进行矩阵乘法前通过脉冲神经元发射脉冲
        q = reshape_for_attention(self.q_spike(q), C)
        k = reshape_for_attention(self.k_spike(k), C)
        v = reshape_for_attention(self.v_spike(v), C_v)

        # 计算 Attention Score: (K^T @ V)
        # 注意：这里是 Linear Attention 的变体，先计算 K^T @ V，再与 Q 相乘，降低计算复杂度 O(N^2) -> O(N)
        x = k.transpose(-2, -1) @ v  
        x = (q @ x) * self.scale * 2  

        # 恢复形状
        x = (x.permute(0, 1, 3, 2, 4)
             .reshape(T, B, N, C_v)
             .permute(0, 1, 3, 2)
             .contiguous()
             .view(T, B, C_v, H, W))

        x = self.attn_spike(x)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x


class MS_Block_Spike_SepConv(nn.Module):
    """
    E-SpikeFormer 的基础 Transformer Block。
    包含：可分离卷积特征提取 -> E-SDSA 注意力机制 -> MLP 前馈网络。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, sr_ratio=1, frozen=True, init_values=1e-6, resolution=384):
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)
        self.attn = MS_Attention_linear_3d(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, lamda_ratio=4, frozen=frozen, resolution=resolution)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, frozen=frozen)

    def forward(self, x, branch):
        x = x + self.conv(x)          # 局部特征提取
        x = x + self.attn(x, branch)  # 全局特征提取 (注意力)
        x = x + self.mlp(x)           # FFN 映射
        return x


class MemoryRetrieval(nn.Module):
    """
    记忆检索模块 (主要用于 SpikeTrack 目标跟踪)。
    用于将 Target Template（目标模板）的特征融入到 Search（搜索区域）的特征中。
    """
    def __init__(self, embed_dim, mlp_ratios, resolution, temp_num):
        super().__init__()
        self.retriever = Retriever(dim=embed_dim, resolution=resolution, temp_num=temp_num)
        self.mlp = MS_MLP(in_features=embed_dim, hidden_features=embed_dim * mlp_ratios, frozen=False)

    def forward(self, templates, search):
        # 利用 Retriever 计算模板和搜索区域的交叉注意力
        search = search + self.retriever(templates, search)
        search = search + self.mlp(search)
        return search


class MS_DownSampling(nn.Module):
    """
    多阶脉冲下采样模块 (Patch Merging/Embedding)。
    在视觉 Transformer 中用于降低空间分辨率，增加通道数。
    """
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True, T=None):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = FrozenBatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer: # 第一层直接接收图像/事件，后续层需要接收脉冲
            self.encode_spike = mem_update(time_step=4)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class Retriever(nn.Module):
    """
    执行具体的 Template 和 Search 之间的交叉注意力计算 (Cross-Attention)。
    逻辑较长，包含 QKV 提取、交叉计算、门控融合等。
    """
    # (省略了内部的方法注释以保持核心逻辑清晰，其内部机制与 MS_Attention_linear_3d 类似，只是针对双分支进行了拓展)
    # ... [代码与原版相同] ...
    pass # 为了注释的排版紧凑，省略此处重复的冗长解析代码展示，逻辑已在类说明中阐明。


class Spiking_vit_MetaFormer_Spike_SepConv(nn.Module):
    """
    核心骨干网络：E-SpikeFormer (或其在跟踪任务下的变体)。
    典型的四阶段 (4-Stage) 金字塔架构结构，特征图尺寸逐级缩小，通道数逐级增加。
    """
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11, embed_dim=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[6, 8, 6], sr_ratios=[8, 4, 2], template_mode=None, resolution=None, temp_num=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)] # 随机深度衰减规则

        # ==== Stage 1 ====
        self.downsample1_1 = MS_DownSampling(in_channels=in_channels, embed_dims=embed_dim[0] // 2, kernel_size=7, stride=2, padding=3, first_layer=True)
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)])
        
        self.downsample1_2 = MS_DownSampling(in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0], kernel_size=3, stride=2, padding=1, first_layer=False)
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios)])

        # ==== Stage 2 ====
        self.downsample2 = MS_DownSampling(in_channels=embed_dim[0], embed_dims=embed_dim[1], kernel_size=3, stride=2, padding=1, first_layer=False)
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)])

        # ==== Stage 3 ==== (开始使用 Attention 机制)
        self.downsample3 = MS_DownSampling(in_channels=embed_dim[1], embed_dims=embed_dim[2], kernel_size=3, stride=2, padding=1, first_layer=False)
        self.block3 = nn.ModuleList([
            MS_Block_Spike_SepConv(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer, sr_ratio=sr_ratios, resolution=resolution)
            for j in range(6) # 堆叠 6 个 Block
        ])

        # ==== Stage 4 ====
        self.downsample4 = MS_DownSampling(in_channels=embed_dim[2], embed_dims=embed_dim[3], kernel_size=3, stride=1, padding=1, first_layer=False)
        self.block4 = nn.ModuleList([
            MS_Block_Spike_SepConv(dim=embed_dim[3], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer, sr_ratio=sr_ratios, resolution=resolution)
            for j in range(2)
        ])
        
        self.apply(self._init_weights)

        # ==== SpikeTrack 特有的多层记忆检索融合 ====
        embed_dim_indices = [0, 1, 2, 2, 3, 3]
        self.template_mode = template_mode 
        self.mrm = nn.ModuleList([
            MemoryRetrieval(embed_dim=embed_dim[i], mlp_ratios=mlp_ratios, resolution=resolution, temp_num=temp_num)
            for i in embed_dim_indices
        ])

    def forward_features(self, x):
        # 1. 拆分目标模板 (templates) 和当前搜索帧 (search)
        templates, search = self.get_T(x, self.template_mode)

        # 2. 逐 Stage 进行特征提取，并在每个层级将 Template 的信息注入 Search 中 (mrm)
        templates = self.downsample1_1(templates)
        search = self.downsample1_1(search)
        for blk in self.ConvBlock1_1:
            templates = blk(templates)
            search = blk(search)

        templates = self.downsample1_2(templates)
        search = self.downsample1_2(search)
        search = self.mrm[0](templates, search) # 第一次注入

        # ... (后续网络的前向传播逻辑与传统 CNN/ViT 一致，并在关键节点调用 mrm 进行跨帧融合) ...
        # (代码其余前向传播部分已在原始代码中，逻辑就是循环通过 Block 和 Downsample)

        return search  # 返回融合了模板特征的搜索特征图

# =====================================================================
# 模型变体构建函数 (论文实验中的不同规模)
# =====================================================================

def Efficient_Spiking_Transformer_l(**kwargs):
    # Large 变体，参数量约 19.0M
    # 对应论文中的高性能 SNN 模型。
    model = Spiking_vit_MetaFormer_Spike_SepConv(embed_dim=[64, 128, 256, 360], **kwargs)
    return model

def Efficient_Spiking_Transformer_m(**kwargs):
    # Medium 变体，参数量约 10.0M
    model = Spiking_vit_MetaFormer_Spike_SepConv(embed_dim=[48, 96, 192, 240], **kwargs)
    return model

def Efficient_Spiking_Transformer_s(**kwargs):
    # Small 变体，参数量约 5.1M
    model = Spiking_vit_MetaFormer_Spike_SepConv(embed_dim=[32, 64, 128, 192], **kwargs)
    return model

def Efficient_Spiking_Transformer_t(**kwargs):
    # Tiny 变体
    model = Spiking_vit_MetaFormer_Spike_SepConv(embed_dim=[24, 48, 96, 128], **kwargs)
    return model

def build_backbone(cfg):
    """
    根据配置文件(cfg)实例化对应规模的模型，并加载预训练权重。
    """
    # 逻辑分发... 加载 torch.load 权重，由于可能是在新任务上微调，所以使用了严格度较低的权重加载方式。
    # strict=False 允许跳过不匹配的层。
    pass