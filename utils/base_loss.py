import numpy as np
# import cupy as np
# def softmax_cross_entropy(x, y):
#     ''' 对输入先进行 softmax 操作后再使用交叉熵求损失 '''
#     # softmax forward
#     x = x - np.max(x)
#     out = np.exp(x) / np.reshape(np.sum(np.exp(x), 1), (x.shape[0], 1))

#     loss, dout = cross_entropy(out, y)

#     diag = np.zeros((dout.shape[0],dout.shape[1],dout.shape[1]))
#     for i in range(diag.shape[0]):
#         diag[i, :, :] = np.diag(out[i])
#     # 计算梯度 dout reshape to N x C x 1 * (diag - out reshape to N x C x 1 @ out reshape to N x 1 x C (N=1 时就相当于 out@out.T)) -> N x C x C (这个矩阵一行是 yi 对每个 x 的导数，一列是每个 y 对 xi 的导数) -> sum -> N x 1 x C = N x C
#     dx = np.sum(dout.reshape(dout.shape[0], -1, 1) * (diag - out.reshape((out.shape[0], -1, 1)) @ out.reshape((out.shape[0], 1, -1))), 1)
#     return loss, dx

def cross_entropy(pred, y):
    ''' 
        交叉熵 
        Args: 
            pred: pred 为 softmax 函数的输出结果（这个函数不进行 softmax 操作）
            y: 正确的标签（标量形式，不是 one-hot 形式）
        Return:
            loss: 损失
            dpred：损失对输入的导数
    '''
    # 就是第 label 类的概率变成 -log 然后加起来
    # 反向传播就是第 label 类的导数变成 -1/pred，其它都是 0
    y = y.astype(np.int)
    # 限制最小值，免得被 1e-253 之类的极端数值爆掉
    pred = np.clip(pred, 1e-10, 1)
    log_pred = -np.log(pred)
    loss = np.sum(log_pred[np.arange(0, pred.shape[0]), y]) / pred.shape[0]
    dpred = np.zeros_like(pred)
    dpred[np.arange(0, pred.shape[0]), y] =  - pred[np.arange(0, pred.shape[0]), y] ** (-1)
    dpred = dpred / pred.shape[0]
    return loss, dpred
    
def hinge_loss(scores, y):
    ''' 
        合页损失
        Args: 
            scores: scores 为最后全连接层的输出结果
            y: 正确的标签（标量形式，不是 one-hot 形式）
        Return:
            loss: 损失
            dscores：损失对输入的导数
    '''
    y = y.astype(np.int)
    # 选出 yi
    score_y = scores[range(y.shape[0]), y]
    # si - yi + 1
    score_plus_1_minus_y = scores + 1 - score_y.reshape((score_y.shape[0], 1))
    loss_array = np.maximum(0, score_plus_1_minus_y)
    loss_array[range(y.shape[0]), y] = 0
    # 除的这个主要是为了让 loss 和 dloss 值变小一点，不影响整个 loss 的分布
    loss = np.sum(loss_array) / (scores.shape[0]*scores.shape[1])

    # 最后一步求和的反向传播
    dscores = np.ones_like(loss_array) / (scores.shape[0]*scores.shape[1])
    # loss_array[range(y.shape[0]), y] = 0 的反向传播
    dscores[range(y.shape[0]), y] = 0
    # maximum 操作的反向传播
    dscores[score_plus_1_minus_y < 0] = 0
    # si - yi + 1 操作的反向传播（除 label 外节点上游 grad 传回来乘 1（不变），label 是一行 grad 的 sum 取反）
    dscores[range(y.shape[0]), y] = -np.sum(dscores, 1)
    return loss, dscores

# def hinge_loss2(scores, y):
#     scores = np.array(
#         [[1.0, 1, 1,1,1],
#         [1,1,1,1,1],
#         [1,1,1,1,1]]
#     )
#     y = np.array([0,1,2])
#     print(scores)
#     print(y)
#     print("-------------------Me Trash-----------------")
#     l,dl = hinge_loss_origin(scores, y)
#     print(l)
#     print(dl)

#     import torch
#     y = torch.tensor(y, dtype=torch.long)
#     scores = torch.tensor(scores, requires_grad=True)
#     loss_fn = torch.nn.MultiMarginLoss()
#     loss = loss_fn(scores, y)
#     loss.backward()
#     print("--------------------Torch-------------------")
#     print(loss.detach().numpy())
#     print(scores.grad.numpy())


#     return loss.detach().numpy(), scores.grad.numpy()


# def hinge_loss_l2(scores, y):
#     ''' SVM 合页损失 '''
#     y = y.astype(np.int)
#     # 选出 yi
#     score_y = scores[range(y.shape[0]), y]
#     # si - yi + 1
#     score_plus_1_minus_y = scores + 1 - score_y.reshape((score_y.shape[0], 1))
#     loss_array = np.maximum(0, score_plus_1_minus_y)
#     loss_array_l2 = loss_array ** 2
#     loss_array_l2[range(y.shape[0]), y] = 0
#     loss = np.sum(loss_array_l2) / len(y)

#     # 最后一步求和的反向传播
#     dscores = np.ones_like(loss_array_l2) / len(y)
#     # loss_array[range(y.shape[0]), y] = 0 的反向传播
#     dscores[range(y.shape[0]), y] = 0
#     # 平方操作的反向传播
#     dscores = dscores * 2 * loss_array
#     # maximum 操作的反向传播
#     dscores[score_plus_1_minus_y < 0] = 0
#     # si - yi + 1 操作的反向传播（除 label 外节点上游 grad 传回来乘 1（不变），label 是一行 grad 的 sum 取反）
#     dscores[range(y.shape[0]), y] = -np.sum(dscores, 1)
#     return loss, dscores