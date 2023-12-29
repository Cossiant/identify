import cv2

model_bin = "../dnn/opencv_face_detector_uint8.pb"
config_text = "../dnn/opencv_face_detector.pbtxt"

names = ['liuzishuo']

def DNN_face_detect_demo(image,recognizer):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # load tensorflow model
    net = cv2.dnn.readNetFromTensorflow(model_bin, config=config_text)
    # image = cv2.imread("../../Image/face/test/WJ7.jfif")
    h = image.shape[0]
    w = image.shape[1]

    # 人脸检测
    blobImage = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blobImage)
    Out = net.forward()

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 绘制检测矩形
    for detection in Out[0,0,:,:]:
        score = float(detection[2])
        objIndex = int(detection[1])
        if score > 0.5:
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h

            # 绘制
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 255), thickness=2)

            for x, y, w, h in face:
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=2)
                # 进行匹配度识别，匹配度不足会有bug
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 10:
                idum = names[idnum]
                confidence = "{0}%".format(round(100 - confidence))
            else:
                idum = "unknown"
                confidence = "{0}%".format(round(100 - confidence))


            #  "people:%.2f" % score
            cv2.putText(image,idum, (int(left), int(top) - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 255), 1)


    cv2.imshow('demo', image)

if __name__ == '__main__':

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("MyFaceLBPHModel.xml")

    cap = cv2.VideoCapture(0)
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        DNN_face_detect_demo(frame,recognizer)
        if ord('q') == cv2.waitKey(10):
            break


    cv2.destroyAllWindows()
    cap.release()