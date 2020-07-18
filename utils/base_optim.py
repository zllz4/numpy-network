class Optimizer(object):
    ''' 基本的优化器，可以实现梯度下降 '''
    def __init__(self, lr, reg=0.01):
        self.lr = lr
        self.reg = reg
    def optim(self, w, dw, add_reg=False):
        if add_reg:
            return w - self.lr * (dw + w * self.reg)
        return w - self.lr * dw