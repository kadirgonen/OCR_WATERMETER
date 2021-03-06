import sys
import cv2
import numpy as np
from skimage import morphology

img = cv2.imread('5.jpg',0)

equ = cv2.equalizeHist(img)

equ = equ[80:145, 44:250]
#cv2.imwrite('res.png',res)
#cv2.imshow('hist',equ)
h,w=equ.shape

for i in range(0, w):
    for j in range(0, h):
        equ[0,0]

#bu yontem iyi olmadi
#equ1 = cv2.medianBlur(equ,5)
#thresh1 = cv2.adaptiveThreshold(equ1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#bu yontemde iyi olmadi
#ret2,thresh1 = cv2.threshold(equ,0,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

ret,thresh1 = cv2.threshold(equ,100,255,cv2.THRESH_BINARY)
cv2.imshow('thresh1 binary',thresh1)


#kernel = np.ones((1,1),np.uint8)
#closingIm = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('closing',closingIm)

#kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(1,2))
#kernel1=cv2.getStructuringElement(cv2.MORPH_CROSS,(1,2))
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,2))
#kernel = np.ones((1,2),np.uint8)
#erosionIm = cv2.dilate(thresh1,kernel)
#kernel = np.ones((2,2),np.uint8)
#erosionIm = cv2.erode(thresh1,kernel)
#kernel = np.ones((1,2),np.uint8)
#erosionIm = cv2.dilate(erosionIm,kernel)
#mask = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
#mask = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1)
#erosionIm = morphology.remove_small_objects(erosionIm, min_size=64, connectivity=1)
#cv2.imshow('erosion',erosionIm)
#erosionIm = cv2.medianBlur(thresh1,3)
#cv2.imshow('erosion_median',erosionIm)

erosionIm2=np.copy(thresh1)
cv2.bitwise_not(thresh1,erosionIm2)
cv2.imshow('erosion_inv',erosionIm2)

k=2;
while(k<22):
    mysize=(20,k)
    i=1;
    while(i<w-1-mysize[0]):
        roifirst=erosionIm2[0:mysize[1], i:i+mysize[0]]
        a=np.sum(roifirst)
        #cv2.imshow('roifirst',roifirst)
        #print 'a', a
        if a==0:
            #i=i+mysize[0]+1
            #print 'i',i
            i
        else:
            roisecond=erosionIm2[0:mysize[1]+1, i:i+mysize[0]]
            b=np.sum(roisecond)
            #print 'b',b
            c=b-a
            if c==0:
              erosionIm2[0:mysize[1], i:i+mysize[0]]=0
              print "burada"
        i=i+20
    k=k+1
cv2.imshow('yukarisisilindi',erosionIm2)

k=1;
while(k<7):
    mysize=(k,10)
    i=1;
    while(i<h-1-mysize[1]):
        #roifirst=erosionIm2[0:mysize[1], i:i+mysize[0]]
        roifirst=erosionIm2[i:i+mysize[1], w-mysize[0]-1:w]
        a=np.sum(roifirst)
        #cv2.imshow('roifirst',roifirst)
        #print 'a', a
        if a==0:
            #i=i+mysize[0]+1
            #print 'i',i
            i
        else:
            roisecond=erosionIm2[i:i+mysize[1], w-mysize[0]-2:w]
            b=np.sum(roisecond)
            #print 'b',b
            c=b-a
            if c==0:
              erosionIm2[i:i+mysize[1], w-mysize[0]-2:w]=0
              print "burada"
        i=i+1
    k=k+1

cv2.imshow('sag_silindi',erosionIm2)



k=2;
while(k<30):
    mysize=(20,k)
    i=1;
    while(i<w-1-mysize[0]):
        #roifirst=erosionIm2[0:mysize[1], i:i+mysize[0]]
        roifirst=erosionIm2[h-mysize[1]-1:h, i:i+mysize[0]]
        a=np.sum(roifirst)
        #cv2.imshow('roifirst',roifirst)
        #print 'a', a
        if a==0:
            #i=i+mysize[0]+1
            #print 'i',i
            i
        else:
            roisecond=erosionIm2[h-mysize[1]-2:h, i:i+mysize[0]]
            b=np.sum(roisecond)
            #print 'b',b
            c=b-a
            if c==0:
              erosionIm2[h-mysize[1]-2:h, i:i+mysize[0]]=0
              print "burada"
        i=i+20
    k=k+1

cv2.imshow('alt_silindi',erosionIm2)

kernel = np.ones((2,2),np.uint8)
#openingIm = cv2.morphologyEx(erosionIm2, cv2.MORPH_CLOSE, kernel)
#kernel = np.ones((2,2),np.uint8)
dilateIm2 = cv2.dilate(erosionIm2,kernel)
#cv2.imshow('erode_son_goruntu',erosionIm2)

son=np.copy(dilateIm2)
cv2.bitwise_not(dilateIm2,son)
cv2.imwrite('5.png',son)
cv2.imshow('son',son)

key = cv2.waitKey(0)
if key == 27:  # (escape to quit)
    cv2.waitKey(500)
    cv2.destroyAllWindows() 
    sys.exit()
