import cv2
import numpy as np

# 测试训练结果
# predict_image = cv2.imread("../../Image/face/test/WJ7.jfif", 0)
# label, confidence = recongizer.predict(predict_image)
#
# print("标签", label)
# print("可信度", confidence)

def kaishixunlian():
    images = []
    for i in range(11):
        i = i + 1
        #对读取到的图片进行编排
        images.append(cv2.imread("../../Image/face/SQL/" + str('liu ') + '(' + str(i) + ').jpg', 0))
    # 对数据进行标定
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 通过算法训练模型
    recongizer = cv2.face.LBPHFaceRecognizer_create(threshold=65)
    # 给数据集添加标签
    recongizer.train(images, np.array(labels))
    # 保存数据集
    recongizer.save("MyFaceLBPHModel.xml")
    print("训练完成")