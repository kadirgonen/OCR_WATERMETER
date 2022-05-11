import sys
import numpy as np
import cv2
import GetRoiofNumber

def tanima(count):
    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]
    i=0
    while (i<count):
        i=i+1
        img=GetRoiofNumber.GetRoi(i)
        im = img.copy()
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

        for cnt in contours:
            if cv2.contourArea(cnt)>40:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if (h>14) & (h<25)& (w>6) & (w<25):
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                    roi = img[y:y+h,x:x+w]
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
                        
    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "training complete"

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)
    cv2.waitKey(500)
    cv2.destroyAllWindows() 
