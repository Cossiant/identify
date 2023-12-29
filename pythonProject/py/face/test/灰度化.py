import cv2
import numpy as np

#光进行二值化处理
def method_1(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    t,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return binary

#先高斯模糊去噪声，然后二值化图像
def method_2(image):
    blurred = cv2.GaussianBlur(image,(3,3),0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGRA2GRAY)
    t,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return binary

#先均值迁移去噪声，然后二值化的图像
def method_3(image):
    blurred = cv2.pyrMeanShiftFiltering(image,10,100)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGRA2GRAY)
    t,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return binary

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread("../../Image/face/test/OIP-C.jfif")
    # 对图像进行尺寸处理
    # resize_image = cv2.resize(image,dsize=(400,400))
    # 打印图像
    cv2.imshow("image", image)

    ret = method_1(image)
    cv2.imshow("1",ret)

    ret = method_2(image)
    cv2.imshow("2", ret)

    ret = method_3(image)
    cv2.imshow("3", ret)

    cv2.waitKey(0)
    cv2.destroyAllWindows()