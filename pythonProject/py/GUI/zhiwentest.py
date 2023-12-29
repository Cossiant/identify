import os
import cv2
import cv2 as cv
from cv2 import *
import math
from skimage import morphology
import numpy as np
from scipy import signal
from tkinter import *
from tkinter.filedialog import *
from PIL import ImageTk, Image
from zhiwenmatch import feature
from zhiwenmatch import pointMatch


# 增强输入的灰度图像的对比度
def contrast(img_gray):
    maxlist = list()
    minlist = list()
    for line in img_gray:
        maxlist.append(max(line))
        minlist.append(min(line))
    M = max(maxlist)
    m = min(minlist)
    N = 250
    n = 50
    h, w = np.shape(img_gray)
    for i in range(h):
        for j in range(w):
            img_gray[i, j] = (N - n) * (int(img_gray[i, j]) - m) / (M - m) + n
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img_gray


# 对灰度图像进行归一化和分割处理，实现对灰度图像的等比例缩放
def normalize(img_gray):
    M0 = 150
    V0 = 2000
    Mi = np.mean(img_gray)
    Vi = np.var(img_gray)
    h, w = np.shape(img_gray)
    print(h, w)
    N = 24  # 图像分割成（N*N）网格
    h1 = int(h / N)  # 分割块的高度
    w1 = int(w / N)  # 分割块的宽度

    for i in range(h):
        for j in range(w):
            value = img_gray[i, j]
            if value > Mi:
                x = M0 + math.sqrt((V0 * (value - Mi) ** 2) / Vi)
            else:
                x = M0 - math.sqrt((V0 * (value - Mi) ** 2) / Vi)
            img_gray[i, j] = x

    for k in range(N):
        for l in range(N):
            block = img_gray[k * h1:(k + 1) * h1, l * w1:(l + 1) * w1]
            # print(np.shape(block))
            M2 = np.mean(block)
            V2 = np.var(block)

            # 方差的平方小于阀值 或者 纯背景 设置成白色
            if V2 < 300 or V2 == 0:
                for i in range(h1):
                    for j in range(w1):
                        block[i, j] = 255
            else:
                Th = M2 / math.sqrt(V2)
                if Th > 60:
                    for i in range(h1):
                        for j in range(w1):
                            block[i, j] = 255

    return img_gray

# 使用3*3模块实现均值滤波
def filtering(im, shape=(3, 3)):
    # 3*3 均值滤波
    temp = np.ones((3, 3), dtype='float')
    temp *= 1.0 / 9.0
    h, w = np.shape(im)
    In = np.zeros((h, w), dtype='uint8')
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            In[i, j] = int(im[i - 1, j - 1] * temp[0, 0] + im[i - 1, j] * temp[0, 1] + im[i - 1, j + 1] * temp[0, 2] \
                           + im[i, j - 1] * temp[1, 0] + im[i, j] * temp[1, 1] + im[i, j + 1] * temp[1, 2] \
                           + im[i + 1, j - 1] * temp[2, 0] + im[i + 1, j] * temp[2, 1] + im[i + 1, j + 1] * temp[2, 2])
    return In

# 二值化 沿脊线方向增强指纹纹路，采用的方法为基于脊线方向场的增强方法。
def direction_bw(im):
    signal.medfilt(im, (3, 3))  # 中值滤波
    im = filtering(im)  # 均值滤波

    h, w = np.shape(im)

    # 判断和处理脊线方向以及切割
    Im = np.zeros((h, w), dtype='uint8')
    Icc = np.ones((h, w), dtype='uint8')
    for i in range(4, h - 4):
        for j in range(4, w - 4):
            sum1 = int(im[i, j - 4]) + im[i, j - 2] + im[i, j + 2] + im[i, j + 4]
            sum2 = int(im[i - 2, j - 4]) + im[i - 1, j - 2] + im[i + 1, j + 2] + im[i + 2, j + 4]
            sum3 = int(im[i - 4, j - 4]) + im[i - 2, j - 2] + im[i + 2, j + 2] + im[i + 4, j + 4]
            sum4 = int(im[i - 4, j - 2]) + im[i - 2, j - 1] + im[i + 2, j + 1] + im[i + 4, j + 2]
            sum5 = int(im[i - 4, j]) + im[i - 2, j] + im[i + 2, j] + im[i + 4, j]
            sum6 = int(im[i - 4, j + 2]) + im[i - 2, j + 1] + im[i + 2, j - 1] + im[i + 4, j - 2]
            sum7 = int(im[i - 4, j + 4]) + im[i - 2, j + 2] + im[i + 2, j - 2] + im[i + 4, j - 4]
            sum8 = int(im[i - 2, j + 4]) + im[i - 1, j + 2] + im[i + 1, j - 2] + im[i + 2, j - 4]
            sumi = [sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8]
            summax = max(sumi)
            summin = min(sumi)
            summ = math.fsum(sumi)
            b = summ / 8
            if summax + summin + 4 * im[i, j] > 3 * b:
                sumf = summin
            else:
                sumf = summax
            if sumf > b:
                Im[i, j] = 128
            else:
                Im[i, j] = 255

    for i in range(h):
        for j in range(w):
            Icc[i, j] *= Im[i, j]

    for i in range(h):
        for j in range(w):
            if (Icc[i, j] == 128):
                Icc[i, j] = 1
            else:
                Icc[i, j] = 0

    return Icc


# 细化 进一步细化，进行骨架化处理。骨架化是一种细化的方法，可以将图像中的线条保留下来，而去除其他部分。
def thin(binary):
    I = signal.medfilt(binary, (3, 3))  # 中值滤波
    cv.waitKey(0)
    cv.destroyAllWindows()
    h, w = np.shape(I)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if I[i, j] == 0:
                if I[i, j - 1] + I[i - 1, j] + I[i + 1, j] >= 3:
                    I[i, j] = 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if I[i, j] == 1:
                if abs(int(I[i, j + 1]) - int(I[i - 1, j + 1])) + abs(int(I[i - 1, j - 1]) - int(I[i - 1, j])) + abs(
                        int(I[i - 1, j]) - int(I[i - 1, j - 1])) \
                        + abs(int(I[i - 1, j - 1]) - int(I[i, j - 1])) + abs(
                    int(I[i, j - 1]) - int(I[i + 1, j - 1])) + abs(int(I[i + 1, j - 1]) \
                                                                   - int(I[i + 1, j])) + abs(
                    int(I[i + 1, j]) - int(I[i + 1, j + 1])) + abs(int(I[i + 1, j + 1]) - int(I[i, j + 1])) != 1:
                    if int((I[i, j + 1] + I[i - 1, j + 1] + I[i - 1, j])) * (
                            I[i, j - 1] + I[i + 1, j - 1] + I[i + 1, j]) + \
                            (I[i - 1, j] + I[i - 1, j - 1] + I[i, j - 1]) * (
                            I[i + 1, j] + I[i + 1, j + 1] + I[i, j + 1]) == 0:
                        I[i, j] = 0

    I = morphology.skeletonize(I)
    h, w = np.shape(I)
    In = np.zeros((h, w), dtype='uint8')
    for i in range(h):
        for j in range(w):
            if not I[i][j]:
                In[i][j] = 255
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if In[i, j] == 0:
                if In[i - 1, j] == 0 and In[i, j + 1] == 0 or In[i - 1, j] == 0 and In[i, j - 1] == 0 \
                        or In[i + 1, j] == 0 and In[i, j - 1] == 0 or In[i + 1, j] == 0 and In[i, j + 1] == 0:
                    In[i, j] = 255
                else:
                    In[i, j] = 0
    return In

# 去除指纹中的空洞和毛刺 如果当前位置点值为0（背景）该点的四邻域点（上下左右）的和大于3则为毛刺，空洞的判断方法为该点为白色（背景）的四周为黑色（前景）八领域点两的和为0，则为空洞。
def clear(im):
    # 去除指纹中的空洞和毛刺
    m, n = np.shape(im)
    Icc = im
    for i in range(m):
        for j in range(n):
            if Icc[i, j] == 1:
                Icc[i, j] = 0
            else:
                Icc[i, j] = 1

    # 去除毛刺
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if Icc[i, j] == 0:
                if Icc[i - 1, j] + Icc[i + 1, j] + Icc[i, j - 1] + Icc[i, j + 1] >= 3:
                    Icc[i, j] = 1
                else:
                    Icc[i, j] = Icc[i, j]
    # 去除空洞
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if Icc[i, j] == 1:
                if abs(int(Icc[i - 1, j - 1]) - int(Icc[i - 1, j])) + abs(int(Icc[i - 1, j]) - int(Icc[i - 1, j + 1])) + \
                        abs(int(Icc[i - 1, j + 1]) - int(Icc[i, j + 1])) + abs(
                    int(Icc[i, j + 1]) - int(Icc[i + 1, j + 1])) \
                        + abs(int(Icc[i + 1, j + 1]) - int(Icc[i + 1, j])) + abs(
                    int(Icc[i + 1, j]) - int(Icc[i + 1, j - 1])) + \
                        abs(int(Icc[i + 1, j - 1]) - int(Icc[i, j - 1])) + abs(
                    int(Icc[i, j - 1]) - int(Icc[i - 1, j - 1])) != 1:
                    if (Icc[i - 1, j - 1] + Icc[i, j - 1] + Icc[i - 1, j]) * (
                            Icc[i + 1, j + 1] + Icc[i + 1, j] + Icc[i, j + 1]) + (
                            Icc[i - 1, j + 1] + Icc[i - 1, j] + Icc[i, j + 1]) \
                            * (Icc[i + 1, j - 1] + Icc[i, j - 1] + Icc[i + 1, j]) == 0:
                        Icc[i, j] = 0

    for i in range(m):
        for j in range(n):
            if Icc[i, j] == 0:
                Icc[i, j] = 1
            else:
                Icc[i, j] = 0
    Icc = thinning(Icc)
    return Icc

# 用开运算和闭运算对图像进行初步细化操作，先腐蚀后膨胀做开运算，先膨胀后腐蚀做闭运算。
def thinning(im):
    # 用开运算和闭运算对图像进行细化操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 定义用于做开闭运算的结构元素
    # 先腐蚀后膨胀做开运算
    im = cv.morphologyEx(im, cv.MORPH_OPEN, kernel)
    # 先膨胀后腐蚀做闭运算
    im = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel)
    return im

#
def fingerTxy():
    img_filename = askopenfilename(initialdir='./DB3_B/', title='选择待识别图片',
                                   filetypes=[("bmp", "*.bmp")])
    #读取图片
    img = cv.imread(img_filename)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #转换为灰度图像
    img_gray = contrast(img_gray)
    #对图片进行处理
    imgnormal = normalize(img_gray)  # 归一化
    binary = direction_bw(imgnormal)  # 二值化
    clean = clear(binary)  # 去除毛刺和空洞
    contour = thin(clean)  # 细化

    cv.waitKey(0)
    cv.destroyAllWindows()
    h, w = np.shape(contour)
    center = (int(h / 2), int(w / 2))
    #使用 feature模块进行特征点提取
    txy1 = feature(contour, center)
    print(txy1)
    matchAll(txy1)
    show(img, oriImg)


def matchAll(txy1):
    #设定对比的文件位置
    directory_name = "SQL"
    for filename in os.listdir(r"../../Image/zhiwen/" + directory_name):
        # img is used to store the image data
        img = cv2.imread("../../Image/zhiwen/SQL/" + filename)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_gray = contrast(img_gray)
        imgnormal = normalize(img_gray)  # 归一化
        binary = direction_bw(imgnormal)  # 二值化
        clean = clear(binary)  # 去除毛刺和空洞
        contour = thin(clean)  # 细化
        cv.waitKey(0)
        cv.destroyAllWindows()
        h, w = np.shape(contour)
        center = (int(h / 2), int(w / 2))
        txy2 = feature(contour, center)  # 提取特征点
        print(txy2)
        result = pointMatch(txy1, txy2)  # 匹配检测
        if result >= 0.9:
            name.insert(END, str(filename) + '匹配成功' + '\n')
            show(img, resultImg)
        elif result < 0.9:
            name.insert(END, str(filename) + '匹配失败' + '\n')


def show(mImg, label):
    global oriTk, resultTk
    mImg = Image.fromarray(mImg)
    if label == oriImg:
        oriTk = ImageTk.PhotoImage(image=mImg)
        label.configure(image=oriTk)
    elif label == resultImg:
        resultTk = ImageTk.PhotoImage(image=mImg)
        label.configure(image=resultTk)


def zhiwenshibie_main():
    # 进行全局变量声明（不声明将找不到变量导致报错）
    global name, oriImg, resultImg
    root = Tk()
    root.title("指纹识别")
    root.geometry('850x500')
    btn1 = Button(root, text="选择指纹并识别", command=fingerTxy)
    btn1.place(x=10, y=10)
    oriImg = Label(root)
    oriImg.place(x=10, y=60)
    resultImg = Label(root)
    resultImg.place(x=350, y=60)
    name = Text(root, height=20, width=22)
    name.place(x=690, y=170)
    root.mainloop()

if __name__ == '__main__':
    root = Tk()
    root.title("指纹识别")
    root.geometry('850x500')
    btn1 = Button(root, text="选择指纹并识别", command=fingerTxy)
    btn1.place(x=10, y=10)
    oriImg = Label(root)
    oriImg.place(x=10, y=60)
    resultImg = Label(root)
    resultImg.place(x=350, y=60)
    name = Text(root, height=20, width=22)
    name.place(x=690, y=170)
    root.mainloop()
