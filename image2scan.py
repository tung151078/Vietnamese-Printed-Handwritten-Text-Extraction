import cv2
import numpy as np
import os

per = 25

imgQ = cv2.imread('phieuthu.jpg')
h,w,c = imgQ.shape
imgQ = cv2.resize(imgQ,(w,h))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'userforms'
myPiclist = os.listdir(path)
print(myPiclist)

for j,y in enumerate(myPiclist):
    img = cv2.imread(path +"/"+y)
    #img = cv2.resize(img,(w//3,h//3))
    #cv2.imshow(y,img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good, None, flags=2)
    imgMatch = cv2.resize(imgMatch,(w//2,h//2))
    cv2.imshow(y+'2',imgMatch)
    
    scrPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, _ = cv2.findHomography(scrPoints,dstPoints, cv2.RANSAC,5.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))
    imgScan = cv2.resize(imgScan,(w,h))
    cv2.imshow(y+'2',imgScan)

cv2.waitKey(0)
