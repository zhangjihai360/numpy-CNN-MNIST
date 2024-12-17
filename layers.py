import numpy as np
import IM2COL



class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = IM2COL.im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = IM2COL.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = IM2COL.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = IM2COL.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

class ReLU:
    """ReLU函数层
    当x<=0时，y=0，当x>0时，y=x"""
    def __init__(self):
        self.mask = None    #设计一个实例变量

    def forward(self, x):
        self.mask = (x <= 0)   #当x小于等于0时保存为True,大于0时保存为False
        out = x.copy()       #复制x赋值为out
        out[self.mask] = 0    #将True的位置赋值为0，其他位置不变

        return out

    def backward(self, dout):
        dout[self.mask] = 0   #原理同上
        dx = dout

        return dx


class Affine:
    """仿射变换，进行的矩阵的乘积运算，几何中，仿射变换包括一次线性变换和一次平移，分别对应神经网络的加权和运算与加偏置运算"""
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """正向传播，y=np.dot(x,w)+b"""
        # 对应张量
        """将输入数组 x 的原始形状存储起来，并将其重新调整为二维的形状，以方便后续的矩阵计算"""
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """反向传播，梯度乘上翻转值的转置矩阵"""
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)   #当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）

        return dx


class Softmax:
    """处理较灵活，能够处理1，2维数组"""
    def __init__(self, x):
        self.x = x

    def forward(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x) # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))


class SoftmaxWithLoss:
    """使用交叉熵误差为Softmax函数计算损失函数"""
    def __init__(self):
        self.loss = None   #损失
        self.y = None    #Softmax函数的输出
        self.t = None   #监督数据，使用one-hot方式编码

    def cross_entropy_error(self, y, t):
        """交叉熵误差"""
        if y.ndim ==1:   #ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度
            t = t.reshape(1, t.size)   #将t矩阵转换为一个1维数组
            y = y.reshape(1, y.size)   #同上

        """处理 t（目标标签）和 y（模型预测值）的形状不一致的情况
        if t.size == y.size:这一条件语句检查t和y的元素总数是否相等。如果它们的size相等，说明t是一个one-hot编码向量。
        t = t.argmax(axis=1):使用argmax(axis=1)会将每一行中最大值的位置作为该样本的类别索引。
        这行代码的目的是从 one-hot编码转换成类别索引，这样t就可以变成一个一维数组，每个元素代表对应样本的类别。"""
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        error = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

        return error

    def forward(self, x, t):
        self.t = t
        softmax = Softmax(x)
        self.y = softmax.forward(x)
        self.loss = self.cross_entropy_error(self.y, self.t)     #使用交叉熵误差计算loss

        return self.loss

    def backward(self, dout=1):
        """使用计算值减去监督值再除以批次的大小即为单个数据的误差"""
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

