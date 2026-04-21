# spikeLearning

一个用于整理和维护脉冲神经网络（Spiking Neural Network, SNN）常用代码的学习型仓库。

目前仓库主要包含两部分内容：

- 脉冲神经元实现：`neuron/ni_lif.py`
- SDT v3 / E-SpikeFormer 风格模型骨架：`model/sdtv3.py`

项目当前更偏向代码积累与结构梳理，适合用于阅读、拆解、二次实验和后续扩展。

## 仓库结构

```text
spikeLearning/
├── README.md
├── neuron/
│   └── ni_lif.py
└── model/
    └── sdtv3.py
```

## 当前内容说明

### 1. `neuron/ni_lif.py`

该文件实现了一个较完整的脉冲神经元基础模块，核心内容包括：

- `Quant`：自定义量化函数，前向中执行截断和离散化，反向中使用代理梯度
- `MultiSpike`：将连续特征映射为多阶脉冲输出
- `mem_update`：带可学习泄漏项的 LIF 风格膜电位更新模块
- `demo_rgb_shape_flow()`：一个可直接运行的最小示例，用于演示输入输出张量形状变化

这个文件适合作为：

- 学习 SNN 神经元前向传播逻辑的入口
- 在自己的模型中复用基础脉冲激活模块
- 调试时间维输入 `[T, B, C, H, W]` 数据流

### 2. `model/sdtv3.py`

该文件整理了一个偏 E-SpikeFormer / SpikeTrack 风格的模型骨架，包含如下模块：

- `SepConv_Spike`：脉冲版可分离卷积模块
- `MS_ConvBlock_spike_SepConv`：局部卷积特征提取模块
- `MS_MLP`：脉冲版前馈网络
- `MS_Attention_linear_3d`：脉冲驱动线性注意力模块
- `MS_Block_Spike_SepConv`：基础 block
- `MS_DownSampling`：分层下采样模块
- `MemoryRetrieval`：模板与搜索区域之间的记忆检索模块
- `Spiking_vit_MetaFormer_Spike_SepConv`：主干网络定义

同时提供了多种模型构造函数：

- `Efficient_Spiking_Transformer_l`
- `Efficient_Spiking_Transformer_m`
- `Efficient_Spiking_Transformer_s`
- `Efficient_Spiking_Transformer_t`

