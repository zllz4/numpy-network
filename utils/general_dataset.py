import os
import glob
import numpy as np
# import cupy as np
import random
from PIL import Image

# import torchvision
# from torchvision import transforms

class GeneralDataset(object):
    ''' 从指定目录读取图片数据建立数据集 '''
    def __init__(self, path, train=True, resize=None, transform=None):
        
        self.transform = transform
        self.train = train
        self._build_from_path(path)
        self.resize = resize
        # print(self.data[19])
        # self.__getitem__(23)
        # self.__len__()

    def _build_from_path(self, path):
        '''
            从指定的文件夹中获取建立数据集所需的类别与图片信息，需要文件夹内部符合一定的层级结构（子文件夹为 class，子文件夹的内容为此 class 的所有 image）
        '''
        train_data = []
        test_data = []
        class_names = []
        class_count = 0
        for item in glob.glob(path+"\\*"):
            if os.path.isdir(item):
                class_name = os.path.basename(item)
                # print(class_name)
                class_names.append(class_name)
                img_count = 0
                for img_path in glob.glob(item+"\\*.jpg"):
                    if img_count % 5 == 0:
                        test_data.append((img_path, class_count))
                    else:
                        train_data.append((img_path, class_count))
                    img_count += 1
                    # if img_count == 5000:
                    #     break
                print("class %s %d images" % (class_name, img_count))
            class_count += 1
        self.classes = class_names
        # for i in range(100):
        #     print(train_data[random.randint(0, len(train_data))])
        # print(self.classes)
        if self.train:
            self.data = train_data
        else:
            self.data = test_data
        print("total %d images,  %d train, %d test" % (len(train_data)+len(test_data), len(train_data), len(test_data)))
        print("[+] build dataset from path %s" % path)

    def __getitem__(self, i):
        img_path, label = self.data[i]
        img = Image.open(img_path)
        # if self.transform:
        #     img = self.transform(img)
        if self.resize is not None:
            img = img.resize(self.resize)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img.reshape((1, img.shape[0], img.shape[1]))
        else:
            img = np.transpose(img, axes=(2,0,1))
        # print(img)
        # print(np.shape(img))
        return img, label

    def __len__(self):
        # print(len(self.data))
        return len(self.data)
        # return 3000

class Mnist(object):
    ''' 读取 mnist 数据集'''
    def __init__(self, path, train=True):
        if train:
            with open(os.path.join(path, "train-images-idx3-ubyte")) as f:
                train_data = np.fromfile(f, np.uint8)
                train_data = train_data[16:]
                train_data = train_data.reshape((60000, 1, 28, 28)).astype(np.float)
                self.data = train_data
            with open(os.path.join(path, "train-labels-idx1-ubyte")) as f:
                train_label = np.fromfile(f, np.uint8)
                train_label = train_label[8:]
                train_label = train_label.reshape((60000,)).astype(np.float)
                self.label = train_label
        else:
            with open(os.path.join(path, "t10k-images-idx3-ubyte")) as f:
                test_data = np.fromfile(f, np.uint8)
                test_data = test_data[16:]
                test_data = test_data.reshape((10000, 1, 28, 28)).astype(np.float)
                self.data = test_data
            with open(os.path.join(path, "t10k-labels-idx1-ubyte")) as f:
                test_label = np.fromfile(f, np.uint8)
                test_label = test_label[8:]
                test_label = test_label.reshape((10000,)).astype(np.float)
                self.label = test_label
        self.classes = ["0","1","2","3","4","5","6","7","8","9"]
    def __getitem__(self, i):
        img = self.data[i]
        label = self.label[i]
        return img.reshape((1, 28, 28)), label

    def __len__(self):
        # print(len(self.data))
        return len(self.data)
        # return 200
    
class DatasetLoader():
    ''' 使用上述的 Dataset 类构建生成器，用作训练/测试数据读取，要求 Dataset 类具有 __len__ 和 __getitem__ 函数'''
    def __init__(self, dataset, batch_size, transform=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = transform
        
    def load_generator(self):
        choices = np.random.choice(len(self.dataset), len(self.dataset), replace=False)
        for i in range(int(len(self.dataset)/self.batch_size)):
            data = []
            label = []
            for j in range(self.batch_size):
                choice = choices[j]
                data.append(self.dataset[int(choice)][0])
                label.append(self.dataset[int(choice)][1])
            choices = np.delete(choices, range(self.batch_size))
            yield (np.stack(data, 0), np.stack(label, 0))
        else:
            data = []
            label = []
            for j in range(len(choices)):
                choice = choices[j]
                data.append(self.dataset[int(choice)][0])
                label.append(self.dataset[int(choice)][1])
            yield (np.stack(data, 0), np.stack(label, 0))
            
