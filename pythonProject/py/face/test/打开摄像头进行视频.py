import cv2

# 现在可以识别出来是个人

def face_detect_demo(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    face_detect = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_alt2.xml")
    face = face_detect.detectMultiScale(gray,1.1,5)
    for x,y,w,h in face:
        cv2.rectangle(image,(x,y),(x+w,y+h),color=(0,255,255),thickness=2)
    cv2.imshow('video_draw',image)

if __name__ == '__main__':
    # 读取图像
    # image = cv2.imread("../../Image/face/test/R.jfif")
    # 对图像进行尺寸处理
    # resize_image = cv2.resize(image, dsize=(400, 400))

    #读取摄像头
    cap = cv2.VideoCapture(0)
    while True:
        flag,frame = cap.read()
        if not flag:
            break
        face_detect_demo(frame);
        if ord('q') == cv2.waitKey(10):
            break

    cv2.destroyAllWindows()
    cap.release()