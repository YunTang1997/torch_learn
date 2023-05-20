import torch
import numpy as np
import typing
import warnings
import matplotlib.pyplot as plt

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
        print(normal_2)  # 这7个数采样分布的均值不同，但是方差都是1
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
        t_0 = torch.cat([t, t], dim=0)  # 按沿着行维度连接张量序列
        t_1 = torch.cat([t, t], dim=1)  # 按沿着列维度连接张量序列
        print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

        # stack
        # 假如数据都是二维矩阵(平面)，它可以把这些一个个平面按第三维(例如：时间序列)压成一个三维的立方体，而立方体的长度就是时间序列长度。
        # 把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
        # stack第一参数为张量序列
        t_stack = torch.stack([t, t, t], dim=2)  # 第一次指定拼接的维度dim=2，结果的维度是[2, 3, 3]，相当于按照高度堆叠
        print("\nt_stack.shape:{}".format(t_stack.shape))
        t_stack = torch.stack([t, t, t], dim=0)  # 原来的tensor已经有了维度0，因此会把tensor往后移动一个维度变为[1,2,3]，再拼接变为[3,2,3]
        print("\nt_stack.shape:{}".format(t_stack.shape))

        # chunk：将张量按照维度dim进行平均切分。若不能整除，则最后一份张量小于其他张量
        a = torch.ones((2, 7))
        a_chunk = torch.chunk(input=a, chunks=3, dim=1)  # 由于7不能整除3，7/3再向上取整是3，因此前两个维度是[2, 3]，所以最后一个切分的张量维度是[2,1]
        for idx, t in enumerate(a_chunk):
            print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

        # split：将张量按照维度dim进行平均切分。可以指定每一个分量的切分长度
        # split_size_or_sections:
        # 为int时，表示每一份的长度，如果不能被整除，则最后一份张量小于其他张量；
        # 为list时，按照list元素作为每一个分量的长度切分。如果list元素之和不等于切分维度(dim) 的值，就会报错。
        b = torch.ones((5, 2))
        b_split = torch.split(tensor=b, split_size_or_sections=[2, 1, 2], dim=0)
        for idx, t in enumerate(b_split):
            print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

        # index_select
        t = torch.randint(0, 9, size=(3, 3))
        # 张量的索引，注意索引的类型不能为float
        idx = torch.tensor([0, 2], dtype=torch.long)
        t_index_select = torch.index_select(input=t, index=idx, dim=0)
        print("t:\n{}\nt_select:\n{}".format(t, t_index_select))

        # masked_select：按照mask中的True进行索引拼接得到一维张量返回
        t = torch.randint(0, 9, size=(3, 3))
        # mask: 与input同形状的布尔类型张量
        mask = t.le(5)
        t_masked_select = torch.masked_select(input=t, mask=mask)
        print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_masked_select))

        # reshape：张量形状重构
        # 当张量在内存中是连续时，返回的张量和原来的张量共享数据内存，改变一个变量时，另一个变量也会被改变。
        t = torch.randperm(8)  # 生成0到8的随机排列[0,8)
        t_reshape = torch.reshape(t, (-1, 2, 2))  # -1代表通过其他维度自行推断
        print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
        # 修改张量 t 的第 0 个元素，张量 t_reshape 也会被改变
        t[0] = 1024
        print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
        print("t.data 内存地址:{}".format(id(t.data)))
        print("t_reshape.data 内存地址:{}".format(id(t_reshape.data)))

        # transpose：交换张量的两个维度。常用于图像的变换，比如把c*h*w变换为h*w*c
        # input：要交换的变量
        # dim0：要交换的第一个维度
        # dim1：要交换的第二个维度
        t = torch.randn(size=(2, 3, 4))
        t_transpose = torch.transpose(input=t, dim0=1, dim1=2)
        print("t shape:{}\nt_transpose shape: {}".format(t.shape, t_transpose.shape))
        # torch.t()：2 维张量转置，对于2维矩阵而言，等价于torch.transpose(input, 0, 1)

        # squeeze：压缩长度为1的维度
        # dim: 若为None，则移除所有长度为1的维度；若指定维度，则当且仅当该维度长度为1时可以移除
        # 维度0和3的长度是 1
        t = torch.rand((1, 2, 3, 1))
        # 可以移除维度0和3
        t_sq = torch.squeeze(t)
        # 可以移除维度0
        t_0 = torch.squeeze(t, dim=0)
        # 不能移除1
        t_1 = torch.squeeze(t, dim=1)
        print("t.shape: {}".format(t.shape))
        print("t: {}".format(t))
        print("t_sq.shape: {}".format(t_sq.shape))  # torch.Size([2, 3])
        print("t_sq: {}".format(t_sq))
        print("t_0.shape: {}".format(t_0.shape))  # torch.Size([2, 3, 1])
        print("t_1.shape: {}".format(t_1.shape))  # torch.Size([1, 2, 3, 1])
        # unsqueeze：根据dim扩展维度，长度为1
        t = torch.randn(size=(2, 3))
        t_unsqueeze = torch.unsqueeze(input=t, dim=0)
        print("t.shape: {}".format(t.shape))  # torch.Size([2, 3])
        print("t: {}".format(t))
        print("t_unsqueeze.shape: {}".format(t_unsqueeze.shape))  # torch.Size([1, 2, 3])
        print("t_unsqueeze: {}".format(t_unsqueeze))

    def tensor_demo_4(self):
        """
        张量的运算

        :return: None
        """
        # add
        t = torch.ones((2, 2))
        t_add = torch.add(input=t, other=t, alpha=2)  # input + alpha * other
        print("t :{}\nt_add: {}".format(t, t_add))

        # addcdiv
        t = torch.ones((2, 2))
        a = torch.randn((2, 2))
        b = torch.addcdiv(input=t, tensor1=a, tensor2=t, value=2)  # input + value * (tensor1 / tensor2)
        print("t :{}\na: {}\nb: {}".format(t, a, b))

        # addcmul
        t = torch.ones((2, 2))
        a = torch.arange(start=1, end=5).reshape((2, 2))
        b = torch.addcmul(input=t, tensor1=t, tensor2=a, value=1)  # input + value * tensor1 * tensor2
        print("t :{}\na: {}\nb: {}".format(t, a, b))

    def tensor_demo_5(self):
        """
        构建简单的一元线性回归

        :return: None
        """
        torch.manual_seed(10)
        x = torch.rand(size=(20, 1)) * 10
        # print(x)
        y = 2 * x + (5 + torch.randn(size=(20, 1))) # torch.randn(size=(20, 1) 标准正态分布模拟噪声
        # print(y)
        w = torch.randn(size=(1,), requires_grad=True)
        # print(w)
        b = torch.zeros(size=(1,), requires_grad=True)
        # print(b)
        lr = 0.05
        for iteration in range(1000):
            # w * x
            wx = torch.mul(w, x)
            # y = w * x + b
            y_pred = torch.add(wx, b)

            # 计算均方误差为损失函数
            loss = (0.5 * (y - y_pred) ** 2).mean()
            # 反向传播
            loss.backward()

            # 更新参数b
            b.data.sub_(lr * b.grad)
            # 更新参数w
            w.data.sub_(lr * w.grad)

            # 更新完之后梯度要归0
            b.grad.zero_()
            w.grad.zero_()

            # 绘图，每隔20次重新绘制直线
            if iteration % 20 == 0:
                plt.scatter(x.data.numpy(), y.data.numpy())
                plt.plot(x.data.numpy(), y_pred.data.numpy(), '-', lw=3)
                plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
                plt.xlim(1.5, 10)
                plt.ylim(8, 28)
                plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
                plt.pause(0.5)
                plt.clf() # 清除图像

                # 如果 MSE 小于 1，则停止训练
                if loss.data.numpy() < 1:
                    break


if __name__ == '__main__':
    tensor_learn = TensorLearn()
    # tensor_learn.tensor_demo_1()
    # tensor_learn.tensor_demo_2()
    # tensor_learn.tensor_demo_3()
    # tensor_learn.tensor_demo_4()
    tensor_learn.tensor_demo_5()
