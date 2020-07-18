from classifier import Classifier
import matplotlib.pyplot as plt
import tkinter as tk 

def train():
    # 创建分类器
    deep_svm = Classifier("traffic sign", "simplenet", "svm")

    # 训练，lr=1e-3
    for i in range(20):
        deep_svm.train_one_epoch(lr=1e-3)
        deep_svm.test(save_path="deep_svm.npy")
    # 恢复最佳参数
    deep_svm.resume("deep_svm.npy")
    # 训练，lr=1e-4
    for i in range(20):
        deep_svm.train_one_epoch(lr=1e-4)
        deep_svm.test(save_path="deep_svm.npy")
    # 恢复最佳参数
    deep_svm.resume("deep_svm.npy")
    # 训练，lr=1e-5
    for i in range(20):
        deep_svm.train_one_epoch(lr=1e-5)
        deep_svm.test(save_path="deep_svm.npy")

    # 创建用于对比的 softmax loss 网络
    deep_softmax = Classifier("traffic sign", "simplenet-softmax", "cross_entropy")
    for i in range(60):
        deep_softmax.train_one_epoch(lr=1e-3)
        deep_softmax.test(save_path="deep_softmax.npy")

    print("svm best acc", deep_svm.best_acc)
    print("softmax best acc", deep_softmax.best_acc)

    # 显示 loss 变化对比 
    plt.figure()
    plt.plot(deep_svm.history["train_epoch"], deep_svm.history["train_loss"])
    plt.plot(deep_svm.history["test_epoch"], deep_svm.history["test_loss"])
    plt.plot(deep_softmax.history["train_epoch"], deep_softmax.history["train_loss"])
    plt.plot(deep_softmax.history["test_epoch"], deep_softmax.history["test_loss"])
    plt.legend(["deep_svm_train_loss","deep_svm_test_loss","deep_softmax_train_loss", "deep_softmax_test_loss"])

    # 显示 acc 变化对比
    plt.figure()
    plt.plot(deep_svm.history["train_epoch"], deep_svm.history["train_acc"])
    plt.plot(deep_svm.history["test_epoch"], deep_svm.history["test_acc"])
    plt.plot(deep_softmax.history["train_epoch"], deep_softmax.history["train_acc"])
    plt.plot(deep_softmax.history["test_epoch"], deep_softmax.history["test_acc"])
    plt.legend(["deep_svm_train_dynamiclr_acc","deep_svm_test_dynamiclr_acc","deep_softmax_train_acc", "deep_softmax_test_acc"])

    # 显示预测结果
    plt.figure()
    deep_svm.eval_25()
    plt.suptitle("pred result of svm loss net")

    # 显示预测结果
    plt.figure()
    deep_softmax.eval_25()
    plt.suptitle("pred result of softmax cross entropy loss net")
    plt.show()

def test():
    # 创建分类器
    deep_svm = Classifier("traffic sign", "simplenet", "svm")
    # 恢复参数
    deep_svm.resume("deep_svm-95.02%.npy")
    # 如果要验证模型的测试集正确率，取消注释此列
    # deep_svm.test()
    # 开启 GUI
    root = tk.Tk()
    root.title('人工智能识别路标')
    root.geometry('100x100')
    testset_eval_button = tk.Button(root, text="预测测试集图像", command=deep_svm.eval_25)
    clipboard_eval_button = tk.Button(root, text="预测剪贴板图像", command=deep_svm.eval_clipboard)
    testset_eval_button.place(x=10, y=10)
    clipboard_eval_button.place(x=10, y=50)
    root.mainloop()  


# train() # 要训练取消注释我
test() # 要测试取消注释我
