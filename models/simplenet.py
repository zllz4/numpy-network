from models.base_layer import *
from utils.base_optim import *
from utils.base_loss import *

class Net(object):
    ''' Net 的基类 '''
    def __init__(self):
        pass
    def __call__(self, x=None):
        return self.forward(x)

class ConvBlock(Net):
    ''' 
        卷积块，结构为 3x3 Conv -> BatchNorm -> ReLU -> (可选的) MaxPool 
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            pool: 是否加池化层
            optimizer: 优化器
            conv_mode: 卷积层是否启用 im2col 加速
            name: 层名
    '''

    def __init__(self, in_channel, out_channel, pool=False, optimizer=Optimizer(1e-5), conv_mode="fast", name="ConvBlock"):
        super(ConvBlock, self).__init__()
        
        self.conv = Conv2D(in_channel, out_channel, kernel_size=3, stride=1, pad=1, optimizer=optimizer, name=name+"-conv", mode="fast")
        self.bn = BatchNorm(in_channel=out_channel, name=name+"-batchnorm", spatial=True, optimizer=optimizer)
        self.relu = ReLU(name=name+"-relu")
        
        
        self.name = name
        
        self.trainable_layer = [self.conv, self.bn]
        self.layer = [self.conv, self.relu, self.bn]
        
        if pool:
            self.pool = MaxPool2D(2,2,name=name+"-pool")
            self.layer.append(self.pool)
        else:
            self.pool = None

    def forward(self, x):
        ''' ConvBlock 块的前向传播 '''
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        if self.pool is not None:
            out = self.pool(out)
        return out

    def backward(self, dout):
        ''' ConvBlock 块的反向传播 '''
        if self.pool is not None:
            dout = self.pool.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.bn.backward(dout)
        dout = self.conv.backward(dout)
        return dout

    def get_weights(self):
        ''' 获取 ConvBlock 块中所有的可训练参数 '''
        weights_dict = {}
        # 对每一层调用 get_weights() 函数
        for layer in self.trainable_layer:
            weights_dict[layer.name] = layer.get_weights()
        return weights_dict

    def set_weights(self, weights_dict):
        ''' 导入 ConvBlock 块中所有的可训练参数 '''
         # 对每一层调用 set_weights() 函数
        for layer in self.trainable_layer:
            layer.set_weights(weights_dict[layer.name])

    def set_mode(self, train):
        ''' 更新 ConvBlock 块中所有层的训练状态 '''
        for layer in self.layer:
            layer.set_mode(train)

class SimpleNet(Net):
    ''' 用于实验的简单网络，结构为 ConvBlock (32核) -> ConvBlock (64核) -> flatten -> Linear -> (可选的) softmax
        Args:
            input_size: 输入大小，注意 input size 不包括 batch size，如果一次输入 128 张 (3,32,32) 
            的图片 input size 就是 (3,32,32)
            classes: 分类数
            optimizer: 优化器
            softmax: 是否加入 softmax 层
    '''
    def __init__(self, input_size, classes, optimizer=Optimizer(0.1), softmax=False):
        # 输入大小
        self.input_size = input_size
        # 分类数
        self.classes = classes
        # 两个卷积块，每个结构为 conv -> relu -> bn -> (pool)
        self.block1 = ConvBlock(self.input_size[0], 32, name="block1", pool=True)
        self.block2 = ConvBlock(32, 64, name="block2", pool=True)
        # 展平
        self.flatten = Flatten(name="flatten")
        # 全连接层
        self.linear1 = Linear(int(64*input_size[1]/4*input_size[2]/4), classes, name="linear1")
        # 可训练层的列表
        self.trainable_layer = [self.block1, self.block2, self.linear1]
        # 网络所有层的列表
        self.layer = [self.block1, self.block2, self.flatten, self.linear1]
        # softmax 层（可选）
        if softmax:
            self.softmax = Softmax(name="softmax")
            self.layer.append(self.softmax)
        else:
            self.softmax = None

    def forward(self, x):
        ''' 前向传播，调用每一层的前向传播函数 '''
        out = self.block1(x)
        out = self.block2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        if self.softmax is not None:
            out = self.softmax(out)
        return out

    def backward(self, dloss):
        ''' 反向传播，调用每一层的反向传播函数 '''
        if self.softmax is not None:
            dout = self.softmax.backward(dloss)
        else:
            dout = dloss
        dout = self.linear1.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.block1.backward(dout)
        return

    def get_weights(self):
        ''' 获取整个网络中所有的可训练参数，调用每一层的 get_weights() 函数 '''
        weights_dict = {}
        for layer in self.trainable_layer:
            # weights_dict[layer.name] = 123
            weights_dict[layer.name] = layer.get_weights()
        return weights_dict

    def set_weights(self, weight_dict):
        ''' 导入整个网络中所有的可训练参数，调用每一层的 set_weights() 函数 '''
        for layer in self.trainable_layer:
            layer.set_weights(weight_dict[layer.name])

    def set_mode(self, train):
        ''' 更新网络中所有层的训练状态，调用每一层的 set_mode() 函数 '''
        for layer in self.layer:
            layer.set_mode(train)

# class SimpleNetOrigin(Net):
#     ''' Conv -> flatten -> Linear '''
#     def __init__(self, input_size, classes, optimizer=Optimizer(0.1), conv_mode="fast", softmax=False):
#         ''' 注意 input size 不包括 batch size，如果一次输入 128 张 (3,32,32) 的图片 input size 就是 (3,32,32) '''
#         self.input_size = input_size
#         self.classes = classes

#         self.block1 = ConvBlock(self.input_size[0], 16, name="block1")
#         self.block2 = ConvBlock(16, 32, name="block2")
#         # self.block3 = ConvBlock(32, 32, pool=False, name="block3")
#         # self.block4 = ConvBlock(32, 64, pool=True, name="block4")
#         # self.block5 = ConvBlock(64, 64, pool=False, name="block5")
        
#         self.flatten = Flatten(name="flatten")
#         self.linear1 = Linear(int(32*input_size[1]*input_size[2]), classes, name="linear1")
#         if softmax:
#             self.softmax = Softmax(name="softmax")
#         else:
#             self.softmax = None
#         # self.trainable_layer = [self.block1, self.block2, self.linear1]
#         self.trainable_layer = [self.block1, self.linear1]
#     def forward(self, x):
#         out = self.block1(x)
#         out = self.block2(out)
#         # out = self.block3(out)
#         # out = self.block4(out)
#         # out = self.block5(out)
#         out = self.flatten(out)
#         out = self.linear1(out)
#         if self.softmax is not None:
#             out = self.softmax(out)
#         return out
#     def backward(self, dloss):
#         if self.softmax is not None:
#             dout = self.softmax.backward(dloss)
#         else:
#             dout = dloss
#         dout = self.linear1.backward(dout)
#         dout = self.flatten.backward(dout)
#         # dout = self.block5.backward(dout)
#         # dout = self.block4.backward(dout)
#         # dout = self.block3.backward(dout)
#         dout = self.block2.backward(dout)
#         dout = self.block1.backward(dout)
#         return
#     def get_weights(self):
#         weights_dict = {}
#         for layer in self.trainable_layer:
#             # weights_dict[layer.name] = 123
#             weights_dict[layer.name] = layer.get_weights()
#         return weights_dict
#     def set_weights(self, weight_dict):
#         print(type(weight_dict))
#         for layer in self.trainable_layer:
#             layer.set_weights(weight_dict[layer.name])
    