import sys
import cv2
import numpy as np

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

############################# testing part  #########################

thresh = cv2.imread('deneme.jpg')
im = cv2.imread('deneme.jpg')
out = np.zeros(thresh.shape,np.uint8)
thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(thresh,255,1,1,9,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>40:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  (h>9) & (h<25)& (w>5) & (w<22):
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            cv2.putText(out,string,(x,y+h),0,1,(0,0,255))
            print string

cv2.imshow('im',im)
cv2.imshow('out',out)
key = cv2.waitKey(0)
if key == 27:  # (escape to quit)
    cv2.waitKey(500)
    cv2.destroyAllWindows() 
    sys.exit()
