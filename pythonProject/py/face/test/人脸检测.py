import cv2

def face_detect_demo(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    face_detect = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_alt2.xml")
    face = face_detect.detectMultiScale(gray,1.1,5)
    # cv2.imshow('gray',gray)
    for x,y,w,h in face:
        cv2.rectangle(image,(x,y),(x+w,y+h),color=(0,255,255),thickness=2)
    cv2.imshow('result2',image)

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread("../../Image/face/test/WJ7.jfif")
    # 对图像进行尺寸处理
    # resize_image = cv2.resize(image, dsize=(400, 400))
    cv2.imshow('result1', image)
    face_detect_demo(image);
    while True:
        if ord('q') == cv2.waitKey(0):
            break
    cv2.destroyAllWindows()