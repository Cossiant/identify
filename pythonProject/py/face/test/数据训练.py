import cv2
import numpy as np

images = []
images.append(cv2.imread("../../Image/face/test/R.jfif", 0))
images.append(cv2.imread("../../Image/face/test/OIP-C.jfif", 0))
images.append(cv2.imread("../../Image/face/test/PT1.jpg", 0))
images.append(cv2.imread("../../Image/face/test/PT2.jfif",0))
images.append(cv2.imread("../../Image/face/test/PT3.jpg",0))
images.append(cv2.imread("../../Image/face/test/PT4.jfif",0))
images.append(cv2.imread("../../Image/face/test/PT5.jpeg", 0))
images.append(cv2.imread("../../Image/face/test/PT6.jfif", 0))
images.append(cv2.imread("../../Image/face/test/WJ1.jfif", 0))
images.append(cv2.imread("../../Image/face/test/WJ2.jpg", 0))
images.append(cv2.imread("../../Image/face/test/WJ3.jfif", 0))
images.append(cv2.imread("../../Image/face/test/WJ4.jfif", 0))
images.append(cv2.imread("../../Image/face/test/WJ5.jpg", 0))
labels = [0, 0, 0, 0,0, 0, 0, 0,1,1,1,1,1]

# 训练模型
recongizer = cv2.face.LBPHFaceRecognizer_create(threshold = 65)

recongizer.train(images, np.array(labels))

recongizer.save("MyFaceLBPHModel.xml")


predict_image = cv2.imread("../../Image/face/test/WJ7.jfif", 0)
label, confidence = recongizer.predict(predict_image)


print("标签", label)

print("可信度", confidence)
