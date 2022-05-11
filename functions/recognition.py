import sys
import cv2
import numpy as np
import GetRoiofNumber as Roi

def recog(imagename):
        #######   training part    ############### 
        samples = np.loadtxt('generalsamples.data',np.float32)
        responses = np.loadtxt('generalresponses.data',np.float32)
        responses = responses.reshape((responses.size,1))

        model = cv2.KNearest()
        model.train(samples,responses)

        img=Roi.GetRoi(imagename)
        #cv2.imshow('im1',img)
        im = img.copy()

        out = np.zeros(img.shape,np.uint8)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = cv2.adaptiveimgold(img,255,1,1,11,2)

        contours,hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
                if cv2.contourArea(cnt)>40:
                        [x,y,w,h] = cv2.boundingRect(cnt)
                        if  (h>10) & (h<25)& (w>7) & (w<22):
                                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                                roi = img[y:y+h,x:x+w]
                                roismall = cv2.resize(roi,(10,10))
                                roismall = roismall.reshape((1,100))
                                roismall = np.float32(roismall)
                                retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
                                string = str(int((results[0][0])))
                                print string
                                cv2.putText(out,string,(x,y+h),0,1,(255,0,0))

        cv2.imshow('im',im)
        cv2.imshow('out',out)
        key = cv2.waitKey(0)
        if key == 27:  # (escape to quit)
            cv2.waitKey(500)
            cv2.destroyAllWindows() 
            sys.exit()


