import cv2
import numpy as np
def RectDetect(Imagename):
    img=cv2.pyrDown(cv2.imread(Imagename,cv2.IMREAD_UNCHANGED))
    imgs=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgs=cv2.Canny(imgs,50,100)
    ret,thresh=cv2.threshold(imgs.copy(),127,255,cv2.THRESH_BINARY)
    image,contours,hier=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    xmax,ymax,wmax,hmax=cv2.boundingRect(contours[0])
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if w>=wmax or h>hmax:
            xmax,ymax,wmax,hmax=x,y,w,h

    wmax=wmax-12;
    hmax=hmax-12;
    xmax=xmax+7;
    ymax=ymax+7;
    Area=np.asarray(img)
    Area=Area[ymax:(ymax+hmax),xmax:(xmax+wmax)]
    return Area

if __name__=="__main__":
    Img=RectDetect('C:\\Users\\Admin\\Desktop\\983428530.jpg')
    cv2.imshow("After processing",Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()