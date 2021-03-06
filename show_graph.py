import numpy as np
import matplotlib.pyplot as plt
import decimal

# create a new context for this task


ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 6


def float_to_str(f):
    d1 = ctx.create_decimal(repr(f))

    return format(d1, 'f')


def cnn():
    modle_name = "XMLCNN"
    cornet_num = "0"
    cornet_dim = "1000"
    bottleneck_dim = "512"

    loss_file = "./loss_{0}_cornet_{1}_bottleneck_dim_{2}.npy".format(modle_name, cornet_num, bottleneck_dim)
    p1_file = "./p1_{0}_cornet_{1}_bottleneck_dim_{2}.npy".format(modle_name, cornet_num, bottleneck_dim)
    p5_file = "./p5_{0}_cornet_{1}_bottleneck_dim_{2}.npy".format(modle_name, cornet_num, bottleneck_dim)

    loss = np.load(loss_file).reshape(30)
    p1 = np.load(p1_file).reshape(30)
    p5 = np.load(p5_file).reshape(30)

    # 字符	颜色
    # ‘b’	蓝色，blue
    # ‘g’	绿色，green
    # ‘r’	红色，red
    # ‘c’	青色，cyan
    # ‘m’	品红，magenta
    # ‘y’	黄色，yellow
    # ‘k’	黑色，black
    # ‘w’	白色，white

    # 字符	类型	字符	类型
    # '-'	实线	'--'	虚线
    # '-.'	虚点线	':'	点线
    # '.'	点	','	像素点
    # 'o'	圆点	'v'	下三角点
    # '^'	上三角点	'<'	左三角点
    # '>'	右三角点	'1'	下三叉点
    # '2'	上三叉点	'3'	左三叉点
    # '4'	右三叉点	's'	正方点
    # 'p'	五角点	'*'	星形点
    # 'h'	六边形点1	'H'	六边形点2
    # '+'	加号点	'x'	乘号点
    # 'D'	实心菱形点	'd'	瘦菱形点
    # '_'	横线点

    title = "{0}_cornet_{1}".format(modle_name, cornet_num)
    epochs = range(1, loss.shape[0] + 1)
    plt.plot(epochs, loss, label="loss")
    for i, j in zip(epochs, loss):
        if i % 5 == 0:
            plt.annotate("{0}".format(float_to_str(j)), xy=(i, j))
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(epochs, p1, 'b', label="p1")
    for i, j in zip(epochs, p1):
        if i % 5 == 0:
            plt.annotate("{0}".format(float_to_str(j)), xy=(i, j))
    plt.plot(epochs, p5, 'g', label="p5")
    for i, j in zip(epochs, p5):
        if i % 5 == 0:
            plt.annotate("{0}".format(float_to_str(j)), xy=(i, j))
    plt.title(title)
    plt.legend()
    plt.show()


def cornet_cnn():
    np.set_printoptions(suppress=True)
    modle_name = "XMLCNN"
    cornet_num = "8"
    cornet_dim = "1000"
    bottleneck_dim = "512"

    loss_file = "./loss_{0}_cornet_{1}_cornet_dim_{2}.npy".format(modle_name, cornet_num, cornet_dim)
    p1_file = "./p1_{0}_cornet_{1}_cornet_dim_{2}.npy".format(modle_name, cornet_num, cornet_dim)
    p5_file = "./p5_{0}_cornet_{1}_cornet_dim_{2}.npy".format(modle_name, cornet_num, cornet_dim)

    loss = np.load(loss_file).reshape(30)
    p1 = np.load(p1_file).reshape(30)
    p5 = np.load(p5_file).reshape(30)

    # 字符	颜色
    # ‘b’	蓝色，blue
    # ‘g’	绿色，green
    # ‘r’	红色，red
    # ‘c’	青色，cyan
    # ‘m’	品红，magenta
    # ‘y’	黄色，yellow
    # ‘k’	黑色，black
    # ‘w’	白色，white

    # 字符	类型	字符	类型
    # '-'	实线	'--'	虚线
    # '-.'	虚点线	':'	点线
    # '.'	点	','	像素点
    # 'o'	圆点	'v'	下三角点
    # '^'	上三角点	'<'	左三角点
    # '>'	右三角点	'1'	下三叉点
    # '2'	上三叉点	'3'	左三叉点
    # '4'	右三叉点	's'	正方点
    # 'p'	五角点	'*'	星形点
    # 'h'	六边形点1	'H'	六边形点2
    # '+'	加号点	'x'	乘号点
    # 'D'	实心菱形点	'd'	瘦菱形点
    # '_'	横线点

    title = "{0}_cornet_{1}_cornet_dim_{2}".format(modle_name, cornet_num, cornet_dim)
    epochs = range(1, loss.shape[0] + 1)
    plt.plot(epochs, loss, label="loss")
    for i, j in zip(epochs, loss):
        if i % 5 == 0:
            plt.annotate("{0}".format(float_to_str(j)), xy=(i, j))
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(epochs, p1, 'b', label="p1")
    for i, j in zip(epochs, p1):
        if i % 5 == 0:
            plt.annotate("{0}".format(round(j, 6)), xy=(i, j))
    plt.plot(epochs, p5, 'g', label="p5")
    for i, j in zip(epochs, p5):
        if i % 5 == 0:
            plt.annotate("{0}".format(j), xy=(i, j))
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    cnn()
    #cornet_cnn()
