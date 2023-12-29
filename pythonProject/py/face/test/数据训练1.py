import cv2
import numpy as np

images = []
for i in range(11):
    i = i+1
    images.append(cv2.imread("../../Image/face/SQL/" + str('liu ') + '(' + str(i) + ').jpg',0))

labels = [0,0,0,0,0,0,0,0,0,0,0]

# 训练模型
recongizer = cv2.face.LBPHFaceRecognizer_create(threshold = 65)

recongizer.train(images, np.array(labels))

recongizer.save("MyFaceLBPHModel.xml")

predict_image = cv2.imread("../../Image/face/test/WJ7.jfif", 0)
label, confidence = recongizer.predict(predict_image)


print("标签", label)

print("可信度", confidence)
