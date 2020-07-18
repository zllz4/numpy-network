import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
from models.simplenet import *
from utils.base_optim import *
from utils.base_loss import *
from utils.general_dataset import *
from PIL import Image
from PIL import ImageGrab

VALID_DATASET = ["emotion", "mnist", "traffic sign"]
VALID_NET = ["simplenet", "simplenet-softmax"]
VALID_LOSS = ["cross_entropy", "svm"]

class Classifier(object):
    ''' 输入指定数据集和模型结构以此定义一个 classifier '''
    def __init__(self, dataset, net, loss):

        assert dataset in VALID_DATASET, 'Invalid dataset name! Valid dataset:' + ' '.join(VALID_DATASET)
        assert net in VALID_NET, 'Invalid net name! Valid net:' + ' '.join(VALID_NET)
        assert loss in VALID_LOSS, 'Invalid loss name! Valid loss:' + ' '.join(VALID_LOSS)

        # 加载数据集
        if dataset == "mnist":
            # 手写数字识别
            path = "mnist"
            train_set = Mnist(path, train=True)
            test_set = Mnist(path, train=False)
            batch_size_train = 256 # 训练集 batch_size
            batch_size_test = 256 # 测试集 batch_size
            input_size = (1,28,28) # 输入图片大小
            classes = train_set.classes # 各个分类的类名
            num_classes = 10 # 分类数
        elif dataset == "emotion":
            # 表情识别
            path = "H://dataset//emotion" # 改成你的 emotion 数据集位置
            train_set = GeneralDataset(path, train=True)
            test_set = GeneralDataset(path, train=False)
            batch_size_train = 64
            batch_size_test = 64
            input_size = (1,48,48)
            classes = train_set.classes
            num_classes = 7
        elif dataset == "traffic sign":
            # 交通标志识别
            path = "H:\\dataset\\GTSRB\\Image"
            train_set = GeneralDataset(path, train=True, resize=(32,32))
            test_set = GeneralDataset(path, train=False, resize=(32,32))
            batch_size_train = 64
            batch_size_test = 64
            input_size = (3,32,32)
            # classes = train_set.classes # 注释掉下面的 classes 赋值行然后取消注释这行采用原本的数字类名（如 00000，00001）
            classes = [
                "Speed limit \n(20km/h)",
                "Speed limit \n(30km/h)",
                "Speed limit \n(50km/h)",
                "Speed limit \n(60km/h)",
                "Speed limit \n(70km/h)",
                "Speed limit \n(80km/h)",
                "End of speed\n limit (80km/h)",
                "Speed limit \n(100km/h)",
                "Speed limit \n(120km/h)",
                "No passing",
                "No passing for\n vechiles over 3.5 \nmetric tons",
                "Right-of-way at \nthe next intersection",
                "Priority road",
                "Yield",
                "Stop",
                "No vechiles",
                "Vechiles over 3.5 \nmetric tons prohibited",
                "No entry",
                "General caution",
                "Dangerous curve to \nthe left",
                "Dangerous curve to \nthe right",
                "Double curve",
                "Bumpy road",
                "Slippery road",
                "Road narrows on the \nright",
                "Road work",
                "Traffic signals",
                "Pedestrians",
                "Children crossing",
                "Bicycles crossing",
                "Beware of ice/snow",
                "Wild animals crossing",
                "End of all speed and \npassing limits",
                "Turn right ahead",
                "Turn left ahead",
                "Ahead only",
                "Go straight or right",
                "Go straight or left",
                "Keep right",
                "Keep left",
                "Roundabout mandatory",
                "End of no passing",
                "End of no passing by \nvechiles over 3.5\n metric tons"
            ]
            num_classes = 43
        
        # 构建网络 
        optimizer = None
        if net == "simplenet":
            optimizer = Optimizer(lr=1e-3, reg=0) # 定义优化器并指定学习率和正则损失系数
            net = SimpleNet(input_size, num_classes, optimizer=optimizer)
        elif net == "simplenet-softmax":
            # 这个比前面那个多了一个 softmax 层
            optimizer = Optimizer(lr=1e-3, reg=0)
            net = SimpleNet(input_size, num_classes, optimizer=optimizer, softmax=True)

        # 指定损失函数
        if loss == "cross_entropy":
            self.loss = cross_entropy
        elif loss == "svm":
            # self.loss = hinge_loss
            self.loss = hinge_loss

        # 数据集读取器
        self.train_iter = DatasetLoader(train_set, batch_size_train)
        self.test_iter = DatasetLoader(test_set, batch_size_test)
        self.eval_iter = DatasetLoader(test_set, 1) # 这个用来进行展示
        
        self.train_batch_size = batch_size_train
        self.test_batch_size = batch_size_test
        self.input_size = input_size
        self.classes = classes
        self.net = net
        self.optimizer = optimizer

        # 显示网络结构
        print("------- Net Sturcture -------")
        net()
        print("-----------------------------")

        print("classes:",classes)
        print("trainset num:",len(train_set.data))
        print("testset num:",len(test_set.data))

        self.epoch = 0
        self.best_acc = 0
        self.history = {}

    def train_one_epoch(self, lr=1e-3):
        ''' 一轮训练 '''
        self.net.set_mode(train=True)

        self.optimizer.lr = lr

        print("epoch:", self.epoch, "lr:", self.optimizer.lr)

        step = 0

        total_loss = 0
        total_true = 0
        total_train = 0

        log_loss = 0
        log_true = 0
        # total_train = 0

        # 每 1 个 step 输出一次训练日志
        record_step = 1

        for inputs, labels in self.train_iter.load_generator():
            # 前向传播
            pred = self.net.forward(inputs)
            # 计算损失
            loss, dloss = self.loss(pred, labels)

            # 获取预测类别
            pred = np.argmax(pred, 1)

            
            # 反向传播
            self.net.backward(dloss)

            step += 1

            total_loss += loss
            total_true += np.sum(pred == labels)
            total_train += len(labels)

            if step % record_step == 0:
                # 计算正确率并输出训练日志
                print("\r[train] step %d loss %.4f acc %.4f      " % (step, total_loss/record_step, total_true*100 / total_train), end="")
                
                log_loss = total_loss/record_step
                log_true = total_true*100 / total_train

                total_loss = 0
                total_true = 0
                total_train = 0
            
            if step == 200:
                break
            
        self.epoch += 1

        self.log("train_epoch", self.epoch)
        self.log("train_loss", log_loss)
        self.log("train_acc", log_true)
        
        self.net.set_mode(train=False)
    
    def test(self, save_path=None):
        ''' 测试 '''
        
        # 在测试集上测试结果
        self.net.set_mode(train=False)
        
        total_acc = 0
        total_test = 0
        total_loss = 0
        step = 0
        
        print()
        for inputs, labels in self.test_iter.load_generator():
            pred = self.net.forward(inputs)
            loss, _ = self.loss(pred, labels)
            pred = np.argmax(pred, 1)
            total_loss += loss
            total_acc += np.sum(pred == labels)
            total_test += len(labels)
            step += 1
            print("\r[test] step %d loss %.4f acc %.4f (%d:%d)       " % (step, total_loss / (step), total_acc * 100 / (total_test), total_acc, total_test), end="")
        print()
        
        test_acc = total_acc * 100 / (total_test)
        test_loss = total_loss / (step)

        # 记录测试日志
        self.log("test_epoch", self.epoch)
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)

        # 如果当前测试正确率高于历史最高，则保存模型参数
        if save_path is not None and test_acc > self.best_acc:
            self.best_acc = test_acc
            np.save(save_path, self.net.get_weights())
            print("model saved...")

    def log(self, name, num):
        ''' 保存日志 '''
        if name in self.history.keys():
            self.history[name].append(num)
        else:
            self.history[name] = [num]

    def resume(self, path):
        ''' 从参数存储文件中恢复模型的参数 '''
        para_dict = np.load(path, allow_pickle=True)
        # net.set_weights() 将递归调用其下每个层的 set_weights() 函数实现参数导入
        # 注意 para_dict 是一个 numpy array，这个 array 里面是一个 dict，要将其变成 dict 需要使用 item()
        self.net.set_weights(para_dict.item())
    
    def eval_clipboard(self):
        ''' 对剪贴板中的图片进行预测 '''
        # 关闭已显示的 figure
        plt.close()
        self.net.set_mode(train=False)
        # 获取剪贴板图片
        image = ImageGrab.grabclipboard()
        if not isinstance(image, Image.Image):
            print("我佛了，剪贴版里没有图像！")
            return
        # resize 操作
        image = image.resize((self.input_size[1], self.input_size[2]))
        # 颜色通道放到第一维度（H x W x C -> C x H x W）
        image = np.transpose(image, (2,0,1))
        # 建立一个新的 figure
        plt.figure()
        # 前向传播
        pred = self.net.forward(image.reshape(1, *image.shape))
        # 获取预测
        pred = np.argmax(pred, 1)
        # 得到预测的标签
        pred_label = self.classes[pred[0]]
        # 把图像的颜色通道变回来
        image = np.transpose(np.reshape(image, self.input_size), (1,2,0))
        # 建立子图
        ax = plt.subplot(1, 1, 1)
        # 展示图片
        if image.shape[2] == 1:
            plt.imshow(image.reshape(image.shape[0], image.shape[1]), cmap="gray")
        else:
            plt.imshow(image)
        # 显示预测的标签
        plt.xlabel("predict:%s" % (pred_label))
        # 删除无关的边框和坐标轴
        plt.xticks([])
        plt.yticks([])
        ax.spines['top'].set_visible(False) 
        ax.spines['bottom'].set_visible(False) 
        ax.spines['left'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        # 显示 figure
        plt.show()

    def eval_25(self):
        ''' 展示测试集中 25 张图片的判断结果 '''
        plt.close()
        self.net.set_mode(train=False)
        for i in range(25):
            image, label = next(self.eval_iter.load_generator())
            pred = self.net.forward(image)
            pred = np.argmax(pred, 1)
            label = self.classes[int(label)]
            # print(classes)
            pred_label = self.classes[pred[0]]
            image = np.transpose(np.reshape(image, self.input_size), (1,2,0))
            ax = plt.subplot(5, 5, i+1)
            if image.shape[2] == 1:
                plt.imshow(image.reshape(image.shape[0], image.shape[1]), cmap="gray")
            else:
                plt.imshow(image)

            # 删除坐标轴但留下 xlabel
            plt.xlabel("correct:%s\npredict:%s" % (label, pred_label))
            plt.xticks([])
            plt.yticks([])
            ax.spines['top'].set_visible(False) 
            ax.spines['bottom'].set_visible(False) 
            ax.spines['left'].set_visible(False) 
            ax.spines['right'].set_visible(False) 
            plt.subplots_adjust(wspace=0, hspace=1.5)
        plt.show()