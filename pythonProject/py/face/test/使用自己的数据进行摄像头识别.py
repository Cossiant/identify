import cv2

names = ["liuzishuo"]
# 现在可以识别出来是个人

def face_detect_demo(image, recognizer):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_detect = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_alt2.xml")
    face = face_detect.detectMultiScale(gray, 1.1, 5)
    # 进行画框
    for x, y, w, h in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=2)
        # 进行匹配度识别，匹配度不足会有bug
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
    # if confidence < 80:
    #     idum = names[idnum]
    #     confidence = "{0}%".format(round(100 - confidence))
    # else:
    #     idum = "unknown"
    #     confidence = "{0}%".format(round(100 - confidence))

        myname = names[idnum]

    cv2.putText(image, str(myname), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(image, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    cv2.imshow('video_draw', image)


if __name__ == '__main__':
    # 读取图像
    # image = cv2.imread("../../Image/face/test/R.jfif")
    # 对图像进行尺寸处理
    # resize_image = cv2.resize(image, dsize=(400, 400))

    # 加载训练的人脸模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("MyFaceLBPHModel.xml")

    # 读取摄像头
    cap = cv2.VideoCapture(0)
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        face_detect_demo(frame, recognizer);
        if ord('q') == cv2.waitKey(10):
            break

    cv2.destroyAllWindows()
    cap.release()
