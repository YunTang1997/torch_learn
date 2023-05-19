import torch
import numpy as np
import typing
import warnings

warnings.filterwarnings("ignore")


class TensorLearn:
    def __int__(self):
        pass

    """
    tensor的属性：
    data：代表被包装的数据
    grad：data的梯度
    grad_fn：创建Tensor的所使用的函数
    requires_grad：张量是否需要计算梯度
    is_leaf：指示是否叶子节点(张量)
    dtype: 张量的数据类型
    shape: 张量的形状
    device: 张量所在设备 (CPU/GPU)
    """
    def tensor_demo_1(self):
        """
        tensor与ndarray之间的转换

        :return: None
        """
        arr = np.ones((3, 3))
        t = torch.tensor(arr, device='cpu')
        print(t)

        # from_numpy会使得新生成的tensor与原来的ndarray共享内存
        arr_1 = torch.from_numpy(arr)
        # print(id(arr) == id(arr_1)) # False
        arr[0, 0] = -1
        print("numpy array: ", arr)
        print("tensor : ", arr_1)

    def tensor_demo_2(self):
        """
        torch.normal的用法

        :return: None
        """
        # 均值与方差均为标量，必须指定size
        normal_1 = torch.normal(mean=0, std=1, size=(4,))
        print("normal_1", "\n", normal_1)
        # mean为标量，std为张量
        # mean为张量，std为标量
        mean = torch.arange(1, 8, dtype=torch.float)
        std = 1
        normal_2 = torch.normal(mean=mean, std=std)
        print("mean:{}\nstd:{}".format(mean, std))
        print(normal_2) # 这7个数采样分布的均值不同，但是方差都是1
        # mean为张量且std也为张量
        mean = torch.arange(1, 5, dtype=torch.float)
        std = torch.arange(1, 5, dtype=torch.float)
        normal_3 = torch.normal(mean=mean, std=std)
        print("mean:{}\nstd:{}".format(mean, std))
        print(normal_3)  # 其中第一个随机数是从normal(1,1)正态分布中采样得到的，其他以此类推

    def tensor_demo_3(self):
        """
        tensor的线性操作

        :return: None
        """
        # cat
        t = torch.ones((2, 3))
        # cat第一参数为张量序列
        t_0 = torch.cat([t, t], dim=0) # 按沿着行维度连接张量序列
        t_1 = torch.cat([t, t], dim=1) # 按沿着列维度连接张量序列
        print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))
        # stack
        # 假如数据都是二维矩阵(平面)，它可以把这些一个个平面按第三维(例如：时间序列)压成一个三维的立方体，而立方体的长度就是时间序列长度。
        # 把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
        # stack第一参数为张量序列
        t_stack = torch.stack([t, t, t], dim=2) # 第一次指定拼接的维度dim=2，结果的维度是[2, 3, 3]，相当于按照高度堆叠
        print("\nt_stack.shape:{}".format(t_stack.shape))
        t_stack = torch.stack([t, t, t], dim=0) # 原来的tensor已经有了维度0，因此会把tensor往后移动一个维度变为[1,2,3]，再拼接变为[3,2,3]
        print("\nt_stack.shape:{}".format(t_stack.shape))
        # chunk
        # 将张量按照维度 dim 进行平均切分。若不能整除，则最后一份张量小于其他张量
        


if __name__ == '__main__':
    tensor_learn = TensorLearn()
    # tensor_learn.tensor_demo_1()
    # tensor_learn.tensor_demo_2()
    tensor_learn.tensor_demo_3()