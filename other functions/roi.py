import sys
import cv2
import numpy as np

img = cv2.imread('1.jpg',0)

ball = img[80:125, 44:250]

#equ = cv2.equalizeHist(img)
#cv2.imwrite('res.png',res)
cv2.imshow('hist',ball)

key = cv2.waitKey(0)
if key == 27:   (escape to quit)
    cv2.waitKey(500)
    cv2.destroyAllWindows() 
    sys.exit()
