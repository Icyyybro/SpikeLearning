import torch
import torch.nn as nn
import torch.nn.functional as F

class Quant(torch.autograd.Function):
    """
    自定义的自动求导函数。
    用于实现脉冲神经元的前向量化（连续变离散）以及反向传播时的代理梯度（Surrogate Gradient）。
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd  # 装饰器：支持PyTorch的自动混合精度(AMP)，加速训练
    def forward(ctx, i, min_value, max_value):
        # 将传入的上下限保存到上下文对象(ctx)中，供反向传播使用
        ctx.min = min_value
        ctx.max = max_value
        # 保存输入张量 i，反向传播时需要根据 i 的值来决定梯度的流向
        ctx.save_for_backward(i)
        
        # 【前向传播核心】：将膜电位 i 限制在 [min_value, max_value] 之间，然后四舍五入取整。
        # 对应论文中的截断和量化操作：clip(U, 0, D)
        result = torch.round(torch.clamp(i, min=min_value, max=max_value))
        return result
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        # 复制上一层传回的梯度
        grad_input = grad_output.clone()
        # 取出前向传播时保存的输入张量 i
        i, = ctx.saved_tensors
        
        # 【反向传播核心】：直通估计器 (Straight-Through Estimator, STE)
        # 如果前向传播时的输入 i 超出了 [min_value, max_value] 的范围，则该神经元处于饱和区，梯度被截断为 0
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        
        # 返回对应于 forward 函数输入的梯度。
        # forward 接收 (i, min_value, max_value) 三个参数，但 min 和 max 不需要梯度，所以返回 None
        return grad_input, None, None

class MultiSpike(nn.Module):
    """
    多阶脉冲神经元模块（Multi-bit Spiking Neuron）。
    将普通的连续特征图转换为多阶（整数）脉冲特征图。
    """
    def __init__(
            self,
            min_value=0,  # 脉冲下限，通常为 0（代表不发射脉冲）
            max_value=4,  # 脉冲上限，对应论文中的 D 阶（例如 D=4 表示允许发射 0, 1, 2, 3, 4 这几种状态）
            Norm=None,    # 归一化系数，用于在输出时缩放特征
    ):
        super().__init__()
        # 如果未指定归一化系数，默认使用最大脉冲值进行归一化，使得最终输出范围在 [0, 1] 之间
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def spike_function(x, min_value, max_value):
        # 静态方法：调用上方自定义的 Quant 类来执行量化和发放
        return Quant.apply(x, min_value, max_value)

    def __repr__(self):
        # 定义 print(model) 时打印出的模块信息
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"

    def forward(self, x):  # x 的形状通常为 B C H W (批次，通道，高，宽)
        # 前向计算：
        # 1. 调用 spike_function 进行多阶脉冲发放，得到整数激活值
        # 2. 除以 self.Norm 进行归一化
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)
    


class mem_update(nn.Module):
    """
    带有可学习泄漏项的 LIF (Leaky Integrate-and-Fire) 脉冲神经元。
    包含了膜电位的积分 (Integrate)、基于 SFA 的多阶发射 (Fire) 和软重置 (Reset) 过程。
    """
    def __init__(self, time_step, Norm=None, skip_ts=False):
        super(mem_update, self).__init__()
        # qtrick 调用了之前定义的 MultiSpike 模块
        # 训练阶段：采用整数替换二值脉冲进行前向传播。
        self.qtrick = MultiSpike(Norm=Norm)  
        self.skip_ts = skip_ts # 是否跳过时间步循环（退化为普通激活函数）
        self.time_step = time_step

        if not self.skip_ts:
            # 初始化可学习的泄漏系数（decay/alpha）。
            # 长度为 time_step - 1，意味着不同时间步之间的泄漏率可以独立学习。
            # 初始值设定为使得 sigmoid(p) = 0.25 的值。
            self.decay = nn.Parameter(torch.full((time_step - 1,), init_sigmoid_param(0.25)))

    def forward(self, x):
        # SNN 中标准的输入张量形状通常是 [T, B, C, H, W]，即时间步 T 在第 0 维。
        transpose_flag = False

        if not self.skip_ts:
            # 如果输入特征将时间步 T 放到了最后一维（例如 temporal-fusion-mlp），
            # 需要将其转置到第 0 维以方便进行时间维度的循环计算。
            if x.shape[0] > 3: 
                x = x.permute(2,0,1)
                transpose_flag = True
                
            output = torch.zeros_like(x) # 用于存储每个时间步发射的脉冲序列
            mem_old = 0                  # 上一时刻的膜电位初始值为 0
            time_window = x.shape[0]     # 获取实际的时间步长度

            # 安全校验：确保时间窗口没有异常偏大
            if time_window > 3:
                raise ValueError("warning! network has bugs, time window dimension > 3 in somewhere.")
                    
            spike = torch.zeros_like(x[0]).to(x.device) # 初始化脉冲张量
            
            # --- 核心：时间步上的循环计算 ---
            for i in range(time_window):
                if i >= 1:
                    # 使用 sigmoid 保证泄漏系数 alpha 始终在 (0, 1) 之间
                    alpha = self.decay[i - 1].sigmoid()
                    # 【积分与重置】(mem_old - spike.detach()) 是减去上一时刻发射的脉冲后的剩余电位（软重置）
                    # 乘以 alpha 进行泄漏衰减，再加上当前时刻的输入 x[i]
                    mem = (mem_old - spike.detach()) * alpha + x[i]
                else:
                    # 第一时刻 i=0 时，没有历史电位积累，直接等于当前输入
                    mem = x[i]
                    
                # 【发射】调用 SFA 方法，生成受限在 0 到 D 之间的整数激活值。
                spike = self.qtrick(mem)
                
                # 记录当前膜电位供下一时刻使用
                mem_old = mem.clone()
                # 记录当前时刻发射的脉冲
                output[i] = spike
                
            # 如果之前转置过，需要将形状还原回去
            if transpose_flag:
                output = output.permute(1,2,0)
        else:
            # 如果配置为 skip_ts，则忽略时间动力学，只执行单次量化（静态任务下的单步 SFA）
            output = self.qtrick(x)
            
        return output

def init_sigmoid_param(p):
    """
    辅助函数：计算目标值 p 的反 sigmoid (Logit)。
    当我们将这个返回值设为 Parameter 的初始值时，网络在应用 sigmoid() 后恰好能得到 p。
    例如：sigmoid(init_sigmoid_param(0.25)) == 0.25
    """
    return torch.log(torch.tensor(p) / (1 - torch.tensor(p)))


def demo_rgb_shape_flow():
    """
    用 mock RGB 图像直接调用 mem_update.forward，演示链路形状变化。
    """
    torch.manual_seed(0)

    batch_size = 2
    channels = 3
    height = 32
    width = 32
    time_step = 3

    # 模拟一批 RGB 图像: [B, C, H, W]
    rgb_image = torch.rand(batch_size, channels, height, width)
    # 将静态图像复制为 3 个时间步，适配当前 mem_update 可直接处理的输入: [T, B, C, H, W]
    temporal_input = rgb_image.unsqueeze(0).repeat(time_step, 1, 1, 1, 1)

    neuron = mem_update(time_step=time_step)

    print("=== Mock RGB -> mem_update shape flow ===")
    print(f"mock rgb_image shape      : {tuple(rgb_image.shape)}")
    print(f"temporal_input shape     : {tuple(temporal_input.shape)}")
    print(f"neuron config            : time_step={neuron.time_step}, skip_ts={neuron.skip_ts}")

    with torch.no_grad():
        output = neuron.forward(temporal_input)

    print(f"\noutput shape             : {tuple(output.shape)}")
    print("\nSummary:")
    print(f"1. 原始 RGB 图像是           [B, C, H, W] = {tuple(rgb_image.shape)}")
    print(f"2. 加上时间维后变成          [T, B, C, H, W] = {tuple(temporal_input.shape)}")
    print(f"3. 直接调用 forward 后，输出仍是 [T, B, C, H, W] = {tuple(output.shape)}")


if __name__ == "__main__":
    demo_rgb_shape_flow()
