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

## 运行环境

建议使用以下基础环境：

- Python 3.10+
- PyTorch 2.x
- timm

可参考：

```bash
pip install torch timm
```

## 快速开始

### 运行脉冲神经元示例

`neuron/ni_lif.py` 自带一个简单 demo，可以直接运行：

```bash
python neuron/ni_lif.py
```

运行后会打印一组 mock RGB 输入在加入时间维前后的形状变化，帮助理解：

- 原始静态图像：`[B, C, H, W]`
- 时间展开后输入：`[T, B, C, H, W]`
- 神经元输出：`[T, B, C, H, W]`

### 在自己的代码中调用 `mem_update`

```python
import torch
from neuron.ni_lif import mem_update

x = torch.randn(3, 2, 3, 32, 32)  # [T, B, C, H, W]
neuron = mem_update(time_step=3)
y = neuron(x)

print(y.shape)
```

## 输入输出约定

当前仓库中的 SNN 模块主要默认以下张量格式：

- 时间维输入：`[T, B, C, H, W]`
- 静态图像：`[B, C, H, W]`

如果输入是静态图像，通常需要先复制或展开到多个时间步，再送入脉冲神经元模块。

## 目前状态

这个仓库还在持续整理中，当前有以下特点：

- `neuron/ni_lif.py` 可以作为独立模块阅读和运行
- `model/sdtv3.py` 更偏向结构化整理与学习参考
- `model/sdtv3.py` 依赖一些外部模块，例如 `lib.models.spiketrack.*`
- 个别类与函数目前仍是骨架或占位实现，例如 `Retriever`、`build_backbone`

因此，当前仓库更适合：

- 阅读和理解 SNN/SpikeFormer 风格模块设计
- 拆分出自己需要的神经元或 block 复用
- 在已有工程中继续补全训练、推理和权重加载逻辑

## 后续可完善方向

后续可以继续补充：

- 完整的训练脚本
- 推理示例
- 数据集组织方式说明
- 预训练权重加载示例
- 单元测试或最小可复现脚本
- 不同模块的论文来源和参考链接

## 适用人群

如果你正在做下面这些事情，这个仓库会比较有帮助：

- 学习脉冲神经网络基础实现
- 阅读多阶脉冲神经元与代理梯度
- 复现 SpikeFormer / SpikeTrack 类模型结构
- 将 SNN 模块迁移到自己的视觉任务中

## License

当前仓库尚未补充许可证信息。如需开源发布，建议后续增加 `LICENSE` 文件并在 README 中明确说明。
