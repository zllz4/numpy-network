import numpy as np
# import cupy as np
from utils.base_optim import *

class Layer(object):
    ''' 所有 Layer 的基类'''
    def __init__(self, name="layer"):
        # 层名
        self.name = name
        # 标志，当处于训练状态时设置为 true，当处于测试状态时设置为 false
        self.train = True
    def __call__(self, x):
        # 重写 __call__ 之前使用前向传播：out = layer.forward(x)
        # 重写 __call__ 之后使用前向传播：out = layer(x) ✔ 更加方便
        return self.forward(x)
    def forward(self, x):
        ''' 前向传播，所有继承自 Layer 类的层都要实现各自的前向传播函数
            Args:
                x : 输入
            Return:
                前向传播结果
        '''
        return x
    def backward(self, dout):
        ''' 反向传播，所有继承自 Layer 类的层都要实现各自的反向传播函数
            Args:
                dout : 输入梯度（一般为后面层反向传播过来的梯度）
            Return:
                输入梯度 dout 对输入 x 的偏导数 dx（作为下一层的输入）
        '''
        return dout
    def get_weights(self):
        ''' 获取本层的所有参数，所有继承自 Layer 类的具有可训练参数的层都要实现各自的 get_weights 函数
            Return:
                应该为 dict，dict 的 key 为参数名，value 为参数值
        '''
        return None
    def set_weights(self, weights_dict):
        ''' 用输入的参数更新本层的所有参数，所有继承自 Layer 类的具有可训练参数的层都要实现各自的 set_weights 函数
            Args:
                weights_dict: 应该为 dict，dict 的 key 为参数名，value 为参数值，与 get_weights() 函数的输出对应
        '''
        pass
    def set_mode(self, train=True):
        ''' 用于递归更新当前训练状态，如 train=True -> train=False
            Args:
                train: bool 值，标志新的训练状态，若为 train 则为 True，若为 test 则为 False
        '''
        self.train = train

class Linear(Layer):
    ''' 全连接层 
        Args:
            in_node: 输入节点数
            out_node: 输出节点数
            weight_scale: 用于确定此层初始化值的大小，此层的初始化参数为 randn() * weight_scale
            name: 层名
            optimizer：使用的优化器
    '''

    def __init__(self, in_node, out_node, weight_scale=1e-3, name="fc_layer", optimizer=Optimizer(0.1)):
        ''' 初始化层 '''
        super(Linear, self).__init__(name=name)

        self.in_node = in_node
        self.out_node = out_node
        self.optimizer = optimizer

        # 参数初始化
        self.weight = np.random.randn(in_node, out_node) * weight_scale
        self.b = np.zeros((out_node,))
        
    def forward(self, x):
        '''
            前向传播
        '''

        # 如果输入为 None 则输出此层的信息
        if x is None:
            print("(%s)\n\tLinear Layer -> in_node=%d\tout_node=%d" % (self.name, self.in_node, self.out_node))
            return None

        # 判断输入是否符合此层的输入维度要求
        assert np.shape(x)[1] == self.in_node, "正向传播输入数据维度不匹配！" 

        # 前向传播
        out = x @ self.weight + self.b

        # 保存此次前向传播的现场，用于反向传播
        self.cache = (x, self.weight, out) 

        return out

    def backward(self, dout):
        '''
            反向传播
        '''

        # 判断输入是否符合此层的输入维度要求
        assert np.shape(dout)[1] == self.out_node, "反向传播输入梯度维度不匹配！"

        # 恢复前向传播的现场
        (x, w, _) = self.cache

        # 反向传播，计算三个偏导
        db = np.sum(dout, 0)
        dw =  x.T @ dout
        dx = dout @ w.T

        # 更新参数
        self.b = self.optimizer.optim(self.b, db)
        self.weight = self.optimizer.optim(self.weight, dw, add_reg=True)

        return dx

    def get_weights(self):
        '''
            获取参数
        '''
        return {"weights":self.weight, "bias":self.b}

    def set_weights(self, weights_dict):
        '''
            导入参数
        '''
        # 判断输入的参数形状是否与本层参数相匹配
        assert weights_dict["weights"].shape == self.weight.shape, self.name + " 层权重参数大小输入不匹配，导入参数失败！应为 " + str(self.weight.shape) + " 实际为 " + str(weights_dict["weights"].shape)
        assert weights_dict["bias"].shape == self.b.shape, self.name + " 层偏置参数大小输入不匹配，导入参数失败！应为 " + str(self.b.shape) + " 实际为 " + str(weights_dict["bias"].shape)
        # 导入参数
        self.weight = weights_dict["weights"]
        self.b = weights_dict["bias"]
        print("%s 层参数导入成功" % self.name)

class ReLU(Layer):
    ''' ReLU 层 '''
    def __init__(self, name="relu_layer"):
        ''' 初始化 '''
        super(ReLU, self).__init__(name=name)
    def forward(self, x):
        ''' 前向传播 '''
        if x is None:
            print("(%s)\n\tReLU Layer" % (self.name))
            return None

        # 真正前向传播部分在这里
        out = np.maximum(0, x)
        
        self.cache = x
        
        return out
    def backward(self, dout):
        ''' 反向传播 '''
        x = self.cache

        # 真正反向传播部分在这里
        dx = dout
        dx[x < 0] = 0

        return dx

class Conv2D(Layer):
    ''' 卷积层 
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核长宽（默认长等于宽）
            stride: 步长
            pad：补零，补零策略为上下左右各补 pad 个零
            weight_scale: 用于确定此层初始化值的大小
            name: 层名
            mode：若为 "fast" 使用速度较快的 im2col 版本卷积，若为 "origin" 则使用原版卷积（速度较慢）
            optimizer: 优化器
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, weight_scale=1e-3, name="conv_layer", mode="origin", optimizer=Optimizer(0.1)):
        ''' 初始化 '''
        super(Conv2D, self).__init__(name=name)

        # 参数初始化
        self.filter_weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * weight_scale
        self.bias = np.zeros((out_channels,))
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.optimizer = optimizer

        assert mode in ["fast", "origin"], 'mode 必须为 "fast" 或 "origin" 之一'
        self.mode = mode # 选择是使用 im2col 快速卷积还是普通的卷积

    def forward(self, x):
        ''' 前向传播 '''
        if self.mode == "fast":  
            # 若 mode == "fast" 调用 im2col 版本的前向传播函数
            out = self.forward_im2col(x)
        elif self.mode == "origin":
            # 若 mode == "origin" 调用 origin 版本的前向传播函数
            out = self.forward_origin(x)
        return out

    def backward(self, dout):
        ''' 反向传播 '''
        if self.mode == "fast":
            dx = self.backward_col2im(dout)
        elif self.mode == "origin":
            dx = self.backward_origin(dout)
        return dx

    def forward_origin(self, x):
        ''' 卷积前向传播，原始版本 '''
        if x is None:
            print("(%s)\n\tConv2D Layer -> in_channel=%d\tout_channel=%d\tkernal_size=%d\tstride=%d\tpad=%d" % (
                self.name, np.shape(self.filter_weight)[1], np.shape(self.filter_weight)[0], self.kernel_size, self.stride, self.pad))
            return None

        # 前向传播部分

        N, C, H, W = x.shape
        w = self.filter_weight
        b = self.bias
        F, _, HH, WW = w.shape
        pad = self.pad
        stride = self.stride
        H_new = int(1 + (H+2*pad-HH) / stride)
        W_new = int(1 + (W+2*pad-WW) / stride)

        # pad
        x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
        x_pad[:, :, pad:-pad, pad:-pad] = x.copy()

        # 初始化输出
        out = np.zeros((N, F, H_new, W_new))

        # 卷积操作：
        # 先 reshape，将输入图片 reshape 成 N x 1 x (C x H_pad x W_pad) 形状
        #              将卷积核 reshape 成 1 x F x (C x H_flt x W_flt) 形状
        # 这样相乘时括号里的部分相乘（输入的图片中 H_pad 和 W_pad 会截取 H_flt 和 W_flt 的大小），括号外的部分广播，得到 N x F x C x H_flt x W_flt
        # 对后三个维度求和，得到 N x F x 1 大小向量
        # 以上操作进行 H_new x W_new 次，得到 N x F x H_new x W_new，此为本次卷积的输出结果

        # reshape
        x_5d = x_pad.reshape((N,1,C,H+2*pad,W+2*pad))
        w_5d = w.reshape((1,F,C,HH,WW))

        # 遍历 (index_y, index_x) 属于 (0 ~ H_new-1, 0 ~ W_new-1)，开始卷积
        for (index_y,index_x) in zip(*np.where(np.ones((H_new,W_new)))):
            # 截取输入 x_5d 中长 H_flt 宽 W_flt 的一部分
            conv_region = x_5d[:,:,:,(stride*index_y):(stride*index_y+HH),(stride*index_x):(stride*index_x+WW)]
            # 卷积操作主要计算
            out[:, :, index_y, index_x] = np.sum(conv_region * w_5d, axis=(2,3,4))
        
        # 加上 bias
        out = out + b.reshape((1, -1, 1, 1))

        # 保存现场
        self.cache = (x, w, b)

        return out

    def backward_origin(self, dout):
        ''' 卷积反向传播，原始版本 '''

        x, w, b = self.cache

        # 反向传播部分

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, H_out, W_out = dout.shape
        pad = self.pad
        stride = self.stride
        dw = np.zeros_like(w)
        dx = np.zeros_like(x)

        # 获取卷积时的 x_pad 
        x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
        dx_pad = x_pad.copy()
        x_pad[:, :, pad:-pad, pad:-pad] = x.copy()

        # 卷积反向传播：
        # 忽略深度，两个 N x F x H_flt x W_flt 的块卷积成一个点（就是前面的 N x F x 1），这里要从这个点还原两个块
        # 由于一个块乘以另一个块得到这个点，所以反过来某个块的梯度就是这个点乘以其中的另一个块

        # reshape，这里的 x_5d 和 w_5d 与卷积时的 x_5d 和 w_5d 相同
        x_5d = x_pad.reshape((N,1,C,H+2*pad,W+2*pad))
        w_5d = w.reshape((1,F,C,HH,WW))

        # 遍历 (index_y, index_x) 属于 (0 ~ H_new-1, 0 ~ W_new-1)，开始卷积的反向传播
        for (index_y,index_x) in zip(*np.where(np.ones((H_out,W_out)))):
            # 获取卷积区域
            x_5d_region = x_5d[:,:,:,(stride*index_y):(stride*index_y+HH),(stride*index_x):(stride*index_x+WW)]
            # 取出卷积的那个点（其实是 N x F 个点，因为一次进行 N x F 次卷积）
            block_dout = dout[:,:,index_y,index_x].reshape((N,F,1,1,1))
            # 得到 dout 对块 x 的梯度
            block_dx = np.sum(w_5d * block_dout, 1)
            # 得到 dout 对块 w 的梯度
            block_dw = np.sum(x_5d_region * block_dout, 0)
            # 总梯度为 n 次卷积反向传播得到的梯度相加（当然位置要对应）
            dw += block_dw
            dx_pad[:,:,(stride*index_y):(stride*index_y+HH),(stride*index_x):(stride*index_x+WW)] += block_dx

        # 得到 dx（dx_pad 去掉 pad 的部分）
        dx = dx_pad[:, :, pad:-pad, pad:-pad]

        # 得到 db
        db = np.sum(dout, (0,2,3))

        # 更新参数
        self.filter_weight = self.optimizer.optim(self.filter_weight, dw, add_reg=True)
        self.bias = self.optimizer.optim(self.bias, db)

        return dx
    def forward_im2col(self, x):
        ''' 卷积前向传播，利用 im2col 加速 '''

        # 前面部分与正常卷积相同
        if x is None:
            print("(%s)\n\tConv2D Layer -> in_channel=%d\tout_channel=%d\tkernal_size=%d\tstride=%d\tpad=%d" % (
                self.name, self.filter_weight.shape[1], self.filter_weight.shape[0], self.kernel_size, self.stride, self.pad))
            return None

        N, C, H_in, W_in = x.shape
        F, _, H_filter, W_filter = self.filter_weight.shape
        x_pad = np.pad(x, ((0,0), (0,0), (self.pad, self.pad), (self.pad,self.pad)), "constant")
        
        assert (H_in + 2*self.pad - H_filter) % self.stride == 0, "H + 2*pad - filter_size 不为 stride 整数倍" 
        assert (W_in + 2*self.pad - W_filter) % self.stride == 0, "W + 2*pad - filter_size 不为 stride 整数倍" 

        H_out = int(1 + (H_in + 2*self.pad - H_filter) / self.stride)
        W_out = int(1 + (W_in + 2*self.pad - W_filter) / self.stride)
       
        # im2col 操作：
        # 取出所有卷积区域（单个卷积区域大小为 C x H_filter x W_filter，一共有 H_out x W_out 个卷积区域，所以所有的卷积区域可以组成一个
        #  N x H_out x W_out x C x H_filter x W_filter 大小的向量，怎么取的牵扯到 np.lib.stride_tricks.as_strided 函数，可以网上搜这
        # 个函数的说明）组成向量 x_col，然后将其后三个维度展平（变成大小为 N x H_out x W_out x (C * H_filter * W_filter) 的四维向量）
        # 同时将 F 个卷积核组成的向量 F x C x H_filter x W_filter 也展平（变成大小为 F x (C * H_filter * W_filter) 的二维向量），然后与
        # 展平后的 x_col 向量做矩阵乘，得到  N x H_out x W_out x (C * H_filter * W_filter) @ (C * H_filter * W_filter) x F = N x H_out x W_out x F，
        # 把第四个维度换到前面来就是卷积的输出 N x F x H_out x W_out

        # np.lib.stride_tricks.as_strided 的 strides 参数选择：
        #   x_pad: N x C x H_in+2*pad x W_in+2*pad 
        #   x_col: N x (H_out x W_out <- 卷积次数) x (C x H_filter x W_filter <- 一次卷积的区域) 
        #       x_col[x,x,x,x,x,i]: stride = base_stride (x_col[x,x,x,x,x,1] -> x_col[x,x,x,x,x,2] 这两个数在 x_pad 上距离为 stride)
        #       x_col[x,x,x,x,i,x]: stride = (W_in+2*self.pad)*base_stride (x_col[x,x,x,x,1,1] -> x_col[x,x,x,x,2,1] 这两个数在 x_pad 上距离为 stride)
        #       x_col[x,x,x,i,x,x]: stride = (H_in+2*pad) * (W_in+2*pad) * base_stride (x_col[x,x,x,1,1,1] -> x_col[x,x,x,2,1,1] 这两个数在 x_pad 上距离为 stride)
        #       x_col[x,x,i,x,x,x]: stride = stride (这个 stride 是指卷积的步长) * base_stride (x_col[x,x,1,1,1,1] -> x_col[x,x,2,1,1,1] 这两个数在 x_pad 上距离为 stride)
        #       x_col[x,i,x,x,x,x]: stride = stride (这个 stride 是指卷积的步长) * (W_in+2*pad) * base_stride (x_col[x,1,1,1,1,1] -> x_col[x,2,1,1,1,1] 这两个数在 x_pad 上距离为 stride)
        #       x_col[i,x,x,x,x,x]: stride = C * (H_in+2*pad) * (W_in+2*pad) * base_stride (x_col[1,1,1,1,1,1] -> x_col[2,1,1,1,1,1] 这两个数在 x_pad 上距离为 stride)
        base_stride = x_pad.strides[-1]
        strides = np.array([
            C * (H_in+2*self.pad) * (W_in+2*self.pad) * base_stride,
            self.stride * (W_in+2*self.pad) * base_stride,
            self.stride * base_stride,
            (H_in+2*self.pad) * (W_in+2*self.pad) * base_stride,
            (W_in+2*self.pad) * base_stride,
            base_stride
        ])
        # 得到卷积区域并展平
        x_col = np.lib.stride_tricks.as_strided(x_pad, shape=(N, H_out, W_out, C, H_filter, W_filter), strides=strides)
        x_col = x_col.reshape((N, H_out, W_out, -1))
        # 卷积核展平
        filter_col = self.filter_weight.reshape((F, -1))
        
        # im2col 卷积（这里就变成普通的矩阵乘加偏置了）
        out = x_col @ filter_col.T + self.bias.reshape((1,1,1,-1))
        # 把最后一个维度挪到前面来
        out = np.transpose(out, axes=(0, 3, 1, 2))

        # 保存现场
        self.cache = (x_col, filter_col, x.shape)

        return out
    def backward_col2im(self, dout):
        ''' 卷积反向传播，利用 col2im 加速 '''
        x_col, filter_col, x_shape = self.cache


        N, F, H_out, W_out = dout.shape
        _, _, H_filter, W_filter = self.filter_weight.shape
        _, C, H_in, W_in = x_shape

        # 得到 db
        db = np.sum(dout, (0,2,3))
        # 前面 “把最后一个维度挪到前面来” 操作的反向操作
        # dout: N x F x H_out x W_out -> N x H_out x W_out x F
        dout = np.transpose(dout, axes=(0,2,3,1))

        # 前面 im2col 卷积操作的反向操作（由于 im2col 卷积操作就是一个全连接层的 forward，这里跟全连接的 backward 是一样的）
        dfilter_col = dout.reshape((-1, F)).T @ x_col.reshape((-1, x_col.shape[-1]))
        # 得到 dw
        dw = dfilter_col.reshape(self.filter_weight.shape) 
        # 得到 dx_col
        dx_col = (dout @ filter_col).reshape((N, H_out, W_out, C, H_filter, W_filter))

        dx_pad = np.zeros((N, C, H_in+2*self.pad, W_in+2*self.pad))
        for index_y in range(H_out):
            for index_x in range(W_out):
                # 把 dx_col 里的梯度值放到合适的位置
                dx_pad[:, :, self.stride*index_y:self.stride*index_y+H_filter, self.stride*index_x:self.stride*index_x+W_filter] += dx_col[:, index_y, index_x, :, :, :]
        
        # 得到 dx
        dx = dx_pad[:, :, self.pad:-self.pad, self.pad:-self.pad]

        # 更新参数
        self.filter_weight = self.optimizer.optim(self.filter_weight, dw, add_reg=True)
        self.bias = self.optimizer.optim(self.bias, db)

        return dx
    def get_weights(self):
        ''' 
            获取参数 
        '''
        return {"weights":self.filter_weight, "bias":self.bias}
    def set_weights(self, weights_dict):
        '''
            导入参数
        '''
        assert weights_dict["weights"].shape == self.filter_weight.shape, self.name + " 层权重参数大小输入不匹配，导入参数失败！应为 " + str(self.filter_weight.shape) + " 实际为 " + str(weights_dict["weights"].shape)
        assert weights_dict["bias"].shape == self.bias.shape, self.name + " 层偏置参数大小输入不匹配，导入参数失败！应为 " + str(self.bias.shape) + " 实际为 " + str(weights_dict["bias"].shape)
        self.filter_weight = weights_dict["weights"]
        self.bias = weights_dict["bias"]
        print("%s 层参数导入成功" % self.name)
        
    


class MaxPool2D(Layer):
    ''' 
        池化层
        Args:
            kernel_size: 卷积核长宽（默认长等于宽）
            stride: 步长
            pad：补零，没实装
            name: 层名

    '''
    def __init__(self, kernel_size=3, stride=1, pad=0, name="pool_layer"):
        super(MaxPool2D, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.stride = stride
        # pad 还没实装
        self.pad = pad
    def forward(self, x):
        if x is None:
            print("(%s)\n\tMaxPool Layer -> kernal_size=%d\tstride=%d\tpad=%d" % (
                self.name, self.kernel_size, self.stride, self.pad))
            return None
        pool_height = self.kernel_size
        pool_width = self.kernel_size
        stride = self.stride
        N, C, H, W = x.shape
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        out = np.zeros((N, C, H_out, W_out))
        for (index_y,index_x) in zip(*np.where(np.ones((H_out,W_out)))):
            out[:,:,index_y,index_x] = np.max(x[:,:,stride*index_y:stride*index_y+pool_height,stride*index_x:stride*index_x+pool_width],(2,3))
        self.cache = x
        return out
    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        x = self.cache
        pool_height = self.kernel_size
        pool_width = self.kernel_size
        stride = self.stride
        N, C, H, W = x.shape
        dx = np.zeros_like(x)
        for (index_y,index_x) in zip(*np.where(np.ones((H_out,W_out)))):
            y_start = stride*index_y
            y_end = stride*index_y+pool_height
            x_start = stride*index_x
            x_end = stride*index_x+pool_height
            mask = (x[:,:,y_start:y_end,x_start:x_end] == np.max(x[:,:,y_start:y_end,x_start:x_end], (2,3)).reshape((N, C, 1, 1))).astype(np.float64)
            dx[:,:,y_start:y_end,x_start:x_end] += dout[:,:,index_y,index_x].reshape((N, C, 1, 1)) * mask
        return dx

class Flatten(Layer):
    ''' 展平层 '''
    def __init__(self, name="flatten_layer"):
        super(Flatten, self).__init__(name=name)
    def forward(self, x):
        if x is None:
            print("(%s)\n\tFlatten Layer" % (self.name))
            return None
        out = np.reshape(x, (np.shape(x)[0], -1))
        self.cache = np.shape(x)
        return out
    def backward(self, dout):
        shape = self.cache
        dx = np.reshape(dout, shape)
        return dx

class Softmax(Layer):
    ''' Softmax 层 '''
    def __init__(self, name="softmax_layer"):
        super(Softmax, self).__init__(name=name)
    def forward(self, x):
        # x: batch_size x class_score
        if x is None:
            print("(%s)\n\tSoftmax Layer" % (self.name))
            return None
        x = x - np.max(x)
        # print("x",x)
        out = np.exp(x) / np.reshape(np.sum(np.exp(x), 1), (x.shape[0], 1))
        self.cache = out
        return out
    def backward(self, dout):
        # 对于 out 的每一行，梯度为 np.sum(diag(out) * out.T @ out, 0)
        # 创建一个多维的对角矩阵
        out = self.cache
        diag = np.zeros((dout.shape[0],dout.shape[1],dout.shape[1]))
        for i in range(diag.shape[0]):
            diag[i, :, :] = np.diag(out[i])
        # print(diag)
        # print(out.reshape((out.shape[0], -1, 1)) @ out.reshape((out.shape[0], 1, -1)))
        # print(dout)
        # print(diag - out.reshape((out.shape[0], -1, 1)) @ out.reshape((out.shape[0], 1, -1)))
        # 计算梯度 dout reshape to N x C x 1 * (diag - out reshape to N x C x 1 @ out reshape to N x 1 x C) -> N x C x C (这个矩阵一行是 yi 对每个 x 的导数，一列是每个 y 对 xi 的导数) -> sum -> N x 1 x C = N x C
        dx = np.sum(dout.reshape(dout.shape[0], -1, 1) * (diag - out.reshape((out.shape[0], -1, 1)) @ out.reshape((out.shape[0], 1, -1))), 1)
        return dx

class BatchNorm(Layer):
    ''' 
        BatchNorm 层，要求输入的 x 大小为 [N, D]（线性层的输出）或 [N, C, H, W]（卷积层的输出） 
        Args:
            in_channel: 通道数（当输入大小为 [N, C, H, W] 时，in_channel = C，当输入大小为 [N, D] 时，in_channel = D）
            name: 层名
            optimizer: 优化器
            momentum: running_mean 和 running_var 的更新动量，running_mean 和 running_var 为测试时用于标准化的 mean 和 var，在训练时 mean 和
             var 依据输入的 batch 计算，同时更新 running_mean / var 为 (1-momentum)*new_mean/var + momentum * old_mean/var，在测试时 running_mean/var 
            不进行更新
            spatial: 若为 True，将输入视为 [N, C, H, W]，也就是对卷积输出进行空间上的 4 维 BatchNorm，若为 False，则将输入视为 [N, D]，也就是对全连接层的输出进行 BatchNorm
    '''
    def __init__(self, in_channel, name, optimizer, momentum=0.9, spatial=False):
        super(BatchNorm, self).__init__(name=name)
        self.gamma = np.ones(in_channel)
        self.beta = np.zeros(in_channel)
        
        self.momentum = momentum
        self.spatial = spatial # 输入是两个维度还是四个维度的图片（后者需要进行 spatial 层面的 batchnorm）
        self.optimizer = optimizer

        self.eps = 1e-5
        self.train = True # 此层在训练与评估阶段有不同的表现

        self.running_mean = np.zeros(in_channel)
        self.running_var = np.zeros(in_channel)

    def batchnorm_forward(self, x):
        ''' BatchNorm 层基本前向传播，输入 x 大小为 [N,D] '''
        out = None

        if self.train:
            # 求整个 batch 的均值（(x[0,:] + x[1,:] + ... )/ batch_size）
            mean = np.mean(x, 0)
            # 求整个 batch 的方差
            variance = np.var(x, 0)
            # 减去均值，除以标准差，得到 x_hat
            x_hat = (x - mean.reshape((1,-1))) / (variance.reshape((1,-1)) + self.eps) ** 0.5

            # out = x_hat * gamma + beta
            out = self.gamma.reshape((1,-1)) * x_hat + self.beta.reshape((1,-1))

            # 保存现场
            self.cache = (x, self.gamma, self.beta, mean, variance, x_hat)

            # 更新 running_mean 和 running_var，这个要在 test mode 下使用
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * variance
        else:
            # 测试模式下的前向传播
            x_hat = (x - self.running_mean.reshape((1,-1))) / (self.running_var.reshape((1,-1)) + self.eps) ** 0.5
            out = self.gamma.reshape((1,-1)) * x_hat + self.beta.reshape((1,-1))

        return out
    def batchnorm_backward(self, dout):
        ''' BatchNorm 层基本反向传播，输入 dout 大小为 [N,D] '''
        x, gamma, beta, mean, variance, x_hat = self.cache

        # gamma 的梯度
        dgamma = np.sum(dout * x_hat, 0)
        # beta 的梯度
        dbeta =  np.sum(dout, 0)
        # x_hat 对 dout 的梯度
        dx_hat = dout * gamma.reshape((1,-1))
        # x 对 x_hat 的梯度
        dxi_of_x_hat = dx_hat / (variance.reshape((1,-1)) + self.eps) ** 0.5
        # mean 对 x_hat 的梯度
        dmean_of_x_hat = np.sum(-dx_hat / (variance.reshape((1,-1)) + self.eps) ** 0.5, 0) 
        # var 对 x_hat 的梯度
        dvar_of_x_hat = np.sum(dx_hat * (-1/2) * ((variance.reshape((1,-1)) + self.eps) ** (-1.5)) * (x - mean.reshape((1, -1))), 0)
        # x 对 mean 的梯度
        dxi_of_mean = 1/x.shape[0] * dmean_of_x_hat * np.ones_like(x)
        # x 对 var 的梯度
        dxi_of_var = 1/x.shape[0] * 2 * (x-mean.reshape((1,-1))) * dvar_of_x_hat.reshape((1,-1))  
        # x 对 dout 的梯度，三个加起来
        dxi = dxi_of_x_hat + dxi_of_mean + dxi_of_var
        # 得到 dx
        dx = dxi

        # 更新参数
        self.gamma = self.optimizer.optim(self.gamma, dgamma)
        self.beta = self.optimizer.optim(self.beta, dbeta)
        return dx

    def forward(self, x):
        ''' 前向传播 '''
        if x is None:
            print("(%s)\n\tBatchNorm Layer -> in_channel=%d\tspatial=%d\t" % (
                self.name, self.gamma.shape[0], self.spatial))
            return None

        # 根据 spatial 的值进行不同的前向传播
        if self.spatial:
            assert len(x.shape) == 4 and x.shape[1] == self.gamma.shape[0], "输入维度不符合要求"
            N, C, H, W = x.shape
            # 先进行维度变换以及 reshape，N x C x H x W 转为 N x H x W x C 然后转为 N*H*W x C 最后利用 N x D 的 batchnorm 函数进行 batchnorm
            x_flat = np.transpose(x, (0,2,3,1)).reshape((-1,C))
            # 利用输入为 N x D 的 batchnorm 函数进行操作
            out = self.batchnorm_forward(x_flat)
            # 输出形状变回来
            out = np.transpose(out.reshape((N,H,W,C)), (0,3,1,2))
            return out
        else:
            assert len(x.shape) == 2 and x.shape[1] == self.gamma.shape[0], "输入维度不符合要求"
            return self.batchnorm_forward(x)

    def backward(self, dout):
        ''' 反向传播 '''

        # 根据 spatial 的值进行不同的前向传播
        if self.spatial:
            assert len(dout.shape) == 4 and dout.shape[1] == self.gamma.shape[0], "输入维度不符合要求"   
            N, C, H, W = dout.shape
            dout = np.transpose(dout, (0,2,3,1)).reshape((-1,C))
            dx = self.batchnorm_backward(dout)
            dx = np.transpose(dx.reshape((N, H, W, C)), (0, 3, 1, 2))
            return dx
        else:
            assert len(dout.shape) == 2 and dout.shape[1] == self.gamma.shape[0], "输入维度不符合要求"
            return self.batchnorm_backward(dout)

    def get_weights(self):
        return {"gamma":self.gamma, "beta":self.beta, "running_mean":self.running_mean, "running_var":self.running_var}
    
    def set_weights(self, weights_dict):
        # print(weights_dict["running_mean"])
        assert weights_dict["gamma"].shape == self.gamma.shape, self.name + " 层 gamma 参数大小输入不匹配，导入参数失败! "
        assert weights_dict["beta"].shape == self.beta.shape, self.name + " 层 beta 参数大小输入不匹配，导入参数失败！"
        assert weights_dict["running_mean"].shape == self.gamma.shape, self.name + " 层 running_mean 参数大小输入不匹配，导入参数失败！"
        assert weights_dict["running_var"].shape == self.gamma.shape, self.name + " 层 running_var 参数大小输入不匹配，导入参数失败！"
        self.gamma = weights_dict["gamma"]
        self.beta = weights_dict["beta"]
        self.running_mean = weights_dict["running_mean"]
        self.running_var = weights_dict["running_var"]
        print("%s 层参数导入成功" % self.name)