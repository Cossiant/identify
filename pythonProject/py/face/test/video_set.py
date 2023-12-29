import cv2

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

video = cv2.VideoCapture(0)
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(size)
while True:
    ret, frame = video.read()
    # 二值化处理
    binary = method_3(frame)
    # 腐蚀
    dst = cv2.erode(binary,kernel=(3,3),iterations=5)
    # 形态梯度
    result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, (3, 3), iterations=3)



    cv2.imshow("A video",result)
    c = cv2.waitKey(1)
    if c == 27:
        break
video.release()
cv2.destroyAllWindows()