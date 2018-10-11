import numpy as np
from os import listdir
import operator
import cv2
from PIL import Image
import math
import matplotlib.pyplot as plt

# 开始程序
# input： none
# output yourNumber：最后得到的结果
def loadproject():
    while(1):
        print("是否开始识别（Y/N）：")
        initial = input()
        if(initial == 'N'):
            break
        else:
            getOnePicture()
            imageToGray()
            # grayToSmaller()
            # canny()
            binaryToStandard()
            grayToBinary()

            # trainData, labelVec, testData, testLabelVec = dataSetClassfication()
            # precisionRateTest(trainData, labelVec, testData, testLabelVec)
            # classfication(testData, trainData, labelVec, num = 25)

# 从视频流中截取一帧图片
# input：视频流
# output：yourImage 需要处理的图片
def getOnePicture():
    cameraCapture = cv2.VideoCapture(0)       # 不知道摄像头设备索引
    fps = 30                      #不知道帧率
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(
        'C:/Users/18139/Desktop/getcamera/first.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'),
        fps, size
    )             # 定义文件放置位置  cv2.VideoWriter_fourcc('I', '4', '2', '0')这里的编码格式也要确定
    success, frame = cameraCapture.read()   # 判断是否取得了有效值   frame应该是某一帧
    numFramesRemaining = 10
    while success and numFramesRemaining > 0:
        cv2.imwrite('C:/Users/18139/Desktop/getcamera/picture1.jpg', frame)
        videoWriter.write(frame)           # 将读取到的帧写入视频
        success, frame = cameraCapture.read()
        numFramesRemaining -= 1
    cameraCapture.release()
    img1 = cv2.imread('C:/Users/18139/Desktop/getcamera/picture1.jpg')          # 需要改图片
    cv2.namedWindow("Image")  # 可加可不加，加上的话一般和imshow之间会定义一些窗口事件处理用函数
    cv2.imshow('Image', img1)  # 显示图片
    if cv2.waitKey(1000) == 27 :
        cv2.destroyAllWindows()  # 释放所有窗口


# 数字图像处理1   转为灰度图
# input：yourImage 从视频流中截取的图片
# output：txt 处理后的灰度化图片
def imageToGray():
    img2 = Image.open('C:/Users/18139/Desktop/getcamera/picture1.jpg').convert("L")
    img2.save('C:/Users/18139/Desktop/getcamera/picture2.bmp')
    img2.show()
    img_array = np.array(img2)
    w, h = img_array.shape
    print(w,h)
    fp = open('C:/Users/18139/Desktop/getcamera/array.txt', 'w')
    for i in img_array:
        fp.write(str(i))
    fp.close()

# 数字图像处理2   利用插值法降低像素   暂时用后面的替代
# input: 灰度化图片
# output：降低后像素的图片
def grayToSmaller():
    image2 = Image.open('C:/Users/18139/Desktop/getcamera/picture2.bmp')
    img_array = np.array(image2)
    a = img_array.shape
    img_array2 = cv2.resize(img_array, (int(a[1] / 1.5), int(a[0] / 1.5)), interpolation=cv2.INTER_AREA)  # 可更改数据调整
    image3 = Image.fromarray(img_array2)
    image3.save('C:/Users/18139/Desktop/getcamera/picture3.bmp')
    image3.show()

# 数字图像处理3 利用canny算子实现边缘检测    高斯滤波  计算梯度值与方向   非极大值抑制（NMS） 双阀值选取（这样精度更高）边缘连接    **************
# input: 降低像素后的图片
# output：边缘检测后的图片
def canny():
    img = plt.imread('C:/Users/18139/Desktop/getcamera/picture3.bmp')

    sigma1 = sigma2 = 1
    sum = 0

    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)  # 生成二维高斯分布矩阵
                                                + (np.square(j - 3) / np.square(sigma2)))) / (
                                         2 * math.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]

    gaussian = gaussian / sum

    # print(gaussian)

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # step1.高斯滤波
    gray = rgb2gray(img)
    W, H = gray.shape
    new_gray = np.zeros([W - 5, H - 5])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # plt.imshow(new_gray, cmap="gray")

    # step2.增强 通过求梯度幅值
    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值

    # plt.imshow(d, cmap="gray")

    # setp3.非极大值抑制 NMS
    W2, H2 = d.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):

            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]

                # 如果Y方向幅度值较大
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果X方向幅度值较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0

    # plt.imshow(NMS, cmap = "gray")

    # step4. 双阈值算法检测、连接边缘
    W3, H3 = NMS.shape
    DT = np.zeros([W3, H3])
    # 定义高低阈值
    TL = 0.2 * np.max(NMS)
    TH = 0.3 * np.max(NMS)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                  or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1

    plt.imshow(DT, cmap="gray")

# 数字图像处理4 A4纸矫正                    **********
# input：顶点提取后的图片
# output：矫正后的图片

# 数字图像处理5 插值法将图像化为标准大小    (需要注意的是，插值法会将二值化图片变成非二值化，要在二值化之前进行）
# input：单个图片
# output：标准图并储存
def binaryToStandard():
    image4 = Image.open('C:/Users/18139/Desktop/getcamera/picture2.bmp')
    img_array = np.array(image4)
    a = img_array.shape
    b = a[0]/32
    c = a[1]/32
    img_array2 = cv2.resize(img_array, (int(a[1] / c), int(a[0] / b)), interpolation=cv2.INTER_AREA)  # 可更改数据调整
    fp = open('C:/Users/18139/Desktop/getcamera/array2.txt', 'w')
    for i in img_array2:
        fp.write(str(i))
    fp.close()
    image5 = Image.fromarray(img_array2)
    image5.save('C:/Users/18139/Desktop/getcamera/picture3.bmp')
    image5.show()

# 数字图像处理6  图像二值化   全阈值（这个简单）
# input：提取后的图像
# output； 二值化图片
def grayToBinary():
    image3 = Image.open('C:/Users/18139/Desktop/getcamera/picture3.bmp')
    img_array = np.array(image3)
    img_array2 = img_array
    for i in range(img_array.shape[0]-1):
        for j in range(img_array.shape[1]-1):
            if (img_array[i][j] >= 100):
                img_array2[i][j] = 255
            else:
                img_array2[i][j] = 0
    fp = open('C:/Users/18139/Desktop/getcamera/array3.txt', 'w')
    for i in img_array2:
        fp.write(str(i))
    fp.close()
    image4 = Image.fromarray(img_array2)
    image4.save('C:/Users/18139/Desktop/getcamera/picture4.bmp')
    image4.show()

# 数字图像处理7  垂直方向分割后水平方向分割（统计方面）      ***
# input：二值化的灰度图
# output：分割后的子图（前期处理会使数字断裂）

# 数字图像处理8 子图进行断裂字符修复（滤波器原理）         ***
# input：分割后子图
# output：修复后的子图

# 数字图像处理9  连通域标记法从左到右分割数字          ***
# input：切割后无断点的子图
# output：多个数字切割后的框图

# 数字图像处理10 切割后的每个数字，分离并进行储存为所需格式      ***
# input：切割后带框图的数字
# output：单个带标签（不是数字标签，是位置标签）的表


# 图像数据矩阵变换为向量
# input：imageFileName 处理后二值化的图片； height 图片高度； weight 图片宽度
# output：imageVec 转化后的行向量
def Mat2Vec(imageFileName, height, weight):
    imageVec = np.zeros((1, height*weight))
    fileread = open(imageFileName)
    for i in range(height):
        linestr = fileread.readline()
        for j in range(weight):
            imageVec[0, 32*i+j] = int(linestr(j))
    return imageVec

# 数据可视化         *****************
# input：trainData 用于训练的数据 ；testData 用于测试的数据； labelVec 数据标签
# output：输出图像
def viewTheData():
    return 0

# 分类并处理标准数据集
# input：filename 数据集地址
# output：trainData 用于训练的数据 ；testData 用于测试的数据； labelVec 数据标签; testLabelVec 测试集标签
def dataSetClassfication():
    height = 32
    weight = 32
    pixels = height*weight       # 这里1024需要换为具体我们数据处理得到的像素点的个数
    print("enter the path to the trainSet:")
    trainSetFileName = input()  # 加入文件名
    print("enter the path to the testSet:")
    testSetFileName = input()
    trainDatalist = listdir(trainSetFileName)
    dataNumber = len(trainDatalist)
    trainData = np.zeros((dataNumber, pixels))
    labelVec = []
    for i in range(dataNumber):
        fileHeadName = trainDatalist[i]
        classNumber = int(fileHeadName.split('_')[0])  # 因为在储存的数据时，文件名第一个字符是具体哪个数字
        labelVec.append(classNumber)
        trainData[i, :] = Mat2Vec(trainSetFileName+'/'+fileHeadName, height, weight)
    testDataList = listdir(testSetFileName)
    testNumber = len(testDataList)
    testData = np.zeros((testNumber, pixels))
    testLabelVec = []
    for i in range(testNumber):
        fileHeadName = testDataList[i]
        classNumber = int(fileHeadName.split('_')[0])
        testLabelVec.append(classNumber)
        testData[1, :] = Mat2Vec(testSetFileName+'/'+fileHeadName, height, weight)
    return trainData, labelVec, testData, testLabelVec

# 准确率测试
# input：trainData 用于训练的数据 ；testData 用于测试的数据； labelVec 数据标签; testLabelVec 测试集标签
# output：precisionRate 准确率
def precisionRateTest(trainData, labelVec, testData, testLabelVec):
    num = 25 # 附近数据个数
    errorCount = 0
    testNumber = len(testLabelVec)
    for i in range(testNumber):
        yourNumber = classfication(testData[i], trainData, labelVec, num)
        print("your number :%d true number :%d" % (yourNumber, testLabelVec[i]))
        if (yourNumber != testLabelVec[i]):
            errorCount += 1
    print("precisionRata: %f%%" %(errorCount/testNumber))

# KNN分类器模型
# input：trainData 训练集；testData 测试集/ yourData 需要分辨的数据； labelVec 数据标签； num 附近数据个数
# output：yourNumber 分类得到的数据
def classfication(testData, trainData, labelVec, num):
    disMat = np.tile(testData, (trainData.shape[0], 1))-trainData
    disMat2 = disMat**2
    disMat3 = disMat2.sum(axis=1)
    disMat4 = disMat3**0.5
    sortDistant = disMat4.argsort()   # 得到索引值
    classCount = {}
    for i in range(num):
        labels = labelVec[sortDistant[i]]
        classCount[labels] = classCount.get(labels, 0) + 1
    sortClaccCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    yourNumber = sortClaccCount[0][0]
    return yourNumber

loadproject()