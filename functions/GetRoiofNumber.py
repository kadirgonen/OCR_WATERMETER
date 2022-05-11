import sys
import cv2
import numpy as np


def GetRoi(cimage):
	img = cv2.imread(str(cimage)+'.jpg',0)
	equ = cv2.equalizeHist(img)
	equ = equ[80:145, 44:250]
	h,w=equ.shape
	ret,thresh1 = cv2.threshold(equ,100,255,cv2.THRESH_BINARY)
	cv2.imshow('thresh1 binary',thresh1)
	
	erosionIm2=np.copy(thresh1)
	cv2.bitwise_not(thresh1,erosionIm2)
	#cv2.imshow('erosion_inv',erosionIm2)
	k=2
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
				  #print "burada"
			i=i+20
		k=k+1
	
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
				  #print "burada"
			i=i+1
		k=k+1

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
				  #print "burada"
			i=i+20
		k=k+1
	
	kernel = np.ones((2,2),np.uint8)
	dilateIm2 = cv2.dilate(erosionIm2,kernel)
	son=np.copy(dilateIm2)
	cv2.bitwise_not(dilateIm2,son)
	cv2.imwrite(str(cimage)+'.png',son)
	cv2.imshow('son',son)

	return son