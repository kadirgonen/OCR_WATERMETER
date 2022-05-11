import sys

import numpy as np
import cv2

countimage=2
samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

thresh = cv2.imread('deneme'+'.jpg')
im = thresh.copy()

thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(thresh,(7,7),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,9,2)
cv2.imshow('bb',thresh)
#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
	if cv2.contourArea(cnt)>40:
		[x,y,w,h] = cv2.boundingRect(cnt)

		if (h>10) & (h<25)& (w>5) & (w<25):
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
			roi = thresh[y:y+h,x:x+w]
			roismall = cv2.resize(roi,(10,10))
			cv2.imshow('norm',im)
			key = cv2.waitKey(0)

			if key == 27:  # (escape to quit)
				cv2.waitKey(500)
				cv2.destroyAllWindows() 
				sys.exit()
			elif key in keys:
				responses.append(int(chr(key)))
				sample = roismall.reshape((1,100))
				samples = np.append(samples,sample,0)

print i
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
cv2.waitKey(500)
cv2.destroyAllWindows() 
