import cv2
import numpy as np

img=cv2.imread('C:\\Users\\Admin\\Desktop\\test.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0) #高斯平滑处理原图像降噪 
edges=cv2.Canny(gray,50,120)
#ret,thre=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
lines=cv2.HoughLinesP(edges,1.0,np.pi/180,100,minLineLength=100,maxLineGap=10)
linex=[]
liney=[]
line=[]
epsilon=10
# 尚未处理：HoughLinesP检测出少于或多于四条直线的情况 **********************
for x in range(0,lines.shape[0]):
    for x1,y1,x2,y2 in lines[x]:
        # 所以呀，要判断水平线还是垂直线了。
        # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        if abs(y1-y2)<=epsilon: # vertical
            y1=y2=min(y1,y2)
            linex.append([x1,y1,x2,y1])
        elif abs(x1-x2)<=epsilon: # horizontal
            x1=x2=min(x1,x2)
            liney.append([x1,y1,x1,y2])

# 从左上角 右上角 左下角 右下角的顺序储存在line中
# 坐标为(x1,y1),(x2,y1),(x1,y2),(x2,y2)
y1=min(linex[0][1],linex[1][1])+5
y2=max(linex[0][1],linex[1][1])-5
x1=min(liney[0][0],liney[1][0])+5
x2=max(liney[0][0],liney[1][0])-5
Area=np.asarray(edges)
Area=Area[y1:y2,x1:x2]
cv2.imshow("after process",Area)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("result.jpg",img)
