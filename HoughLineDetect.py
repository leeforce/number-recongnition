import cv2
import numpy as np
import math

img=cv2.imread('C:\\Users\\Admin\\Desktop\\717550502.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0) #高斯平滑处理原图像降噪 
edges=cv2.Canny(gray,50,120)
#ret,thre=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
lines=cv2.HoughLinesP(edges,1.0,np.pi/180,100,minLineLength=100,maxLineGap=10)
linex=[]
liney=[]
line=[]
epsilon=0.4 # tan15°=0.27
# 尚未处理：HoughLinesP检测出少于或多于四条直线的情况 **********************
# 一般来说，出现多于四条直线的情况是因为摄像头距离屏幕过近或者分辩率过高，导致直线的厚度不可被忽视
# 这种情况下有两种解决方案：1.调整好摄像头与屏幕的距离，2.对图像进行reshape操作，减小其尺寸
#while lines.shape[0]>4:
#    # TODO：***************  减小图像分辨率
#    lines=cv2.HoughLinesP(edges,1.0,np.pi/180,100,minLineLength=100,maxLineGap=10)
#    if lines.shape[0]=4:
#        break
# 鲁棒性解决方案 *************** 不改变分辨率，长度那里要根据你的定好的距离调
if lines.shape[0]>4:
    for x in range(0,lines.shape[0]):
        for x1,y1,x2,y2 in lines[x]:
            if x1==x2:
                if abs(y1-y2)<=100:
                    break
            elif y1==y2:
                if abs(x1-x2)<=100:
                    break
            elif abs((y1-y2)/(x1-x2))>epsilon and abs((x1-x2)/(y1-y2))> epsilon: # 说明不是较垂直线或者较水平线
                break
            elif abs((y1-y2)/(x1-x2))<=epsilon and abs((x1-x2))<=150: # 说明线段较短
                break
            elif abs((x1-x2)/(y1-y2))<=epsilon and abs((y1-y2))<=150: # 说明线段较短
                break
            if abs((y1-y2)/(x1-x2))<=epsilon: # vertical
                y1=y2=min(y1,y2)
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                linex.append([x1,y1,x2,y1])
            elif abs((x1-x2)/(y1-y2))<=epsilon: # horizontal
                x1=x2=min(x1,x2)
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                liney.append([x1,y1,x1,y2])
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    Area=img
else:
    for x in range(0,lines.shape[0]):
        for x1,y1,x2,y2 in lines[x]:
        # 所以呀，要判断水平线还是垂直线了。
        # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            if abs((y1-y2)/(x1-x2))<=epsilon: # vertical
                y1=y2=min(y1,y2)
                linex.append([x1,y1,x2,y1])
            elif abs((x1-x2)/(y1-y2))<=epsilon: # horizontal
                x1=x2=min(x1,x2)
                liney.append([x1,y1,x1,y2])

    if len(listx)<2 or len(listy)<2:
        # 令人担心的事情还是发生了 有至少一条边的缺失
        # 需要事先知道在该距离下A4纸的像素长和像素宽
        # 否则根本做不了
        None
    y1=min(linex[0][1],linex[1][1])+5
    y2=max(linex[0][1],linex[1][1])-5
    x1=min(liney[0][0],liney[1][0])+5
    x2=max(liney[0][0],liney[1][0])-5
# 从左上角 右上角 左下角 右下角的顺序储存在line中
# 坐标为(x1,y1),(x2,y1),(x1,y2),(x2,y2)
    Area=np.asarray(edges)
    Area=Area[y1:y2,x1:x2]
cv2.imshow("after process",Area)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("result.jpg",img)
