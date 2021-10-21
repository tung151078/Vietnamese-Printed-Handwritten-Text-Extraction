import cv2
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
from numpy.lib.function_base import append
from re import match
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from pdf2image import convert_from_path
from vietnam_number import w2n, n2w


roi = [[(343, 180), (376, 205), 'text', 'ngay'], 
        [(419, 184), (448, 206), 'text', 'thang'], 
        [(483, 177), (565, 207), 'text', 'nam'], 
        [(230, 205), (752, 243), 'text', 'nguoi_nop'], 
        [(121, 243), (765, 271), 'text', 'dia_chi'], 
        [(145, 273), (751, 304), 'text', 'ly_do_nop'], 
        [(122, 305), (364, 335), 'text', 'tien_so'],    
        [(501, 305), (751, 335), 'text', 'tien_chu_1'], 
        [(57, 330), (751, 370), 'text', 'tien_chu']]

per = 25
pixelThreshold = 500
max_feature = 5000
path = 'samples'

def pre_processing(image):
    image_gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (9, 9), 1)
    th1 = cv2.adaptiveThreshold(image_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    ret1, th2 = cv2.threshold(image_gray, 157, 255, cv2.THRESH_BINARY)
    ret2, th3 = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th3, (9,9),1)
    ret3, th4 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    or_image = cv2.bitwise_or(th4, closing)
    return or_image

#========== Config to use model vietocr ==========
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
# config['weights'] = 'Model_train/weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False

detector = Predictor(config)

#========== Alignment Image for sample form ===========
# Detect feature
img_template = cv2.imread('Sample.png')
h,w,c = img_template.shape
im1Gray = cv2.cvtColor(img_template,cv2.COLOR_RGB2GRAY) 

orb = cv2.ORB_create(max_feature)
kp1, des1 = orb.detectAndCompute(im1Gray,None)

path = 'samples'
myPiclist = os.listdir(path)

print(myPiclist)
# check file .pdf

for file in os.listdir(path):
    if fnmatch.fnmatch(file, '*.pdf'):      
        pages = convert_from_path(path + "/" + file, 200, poppler_path=r'C:\Program Files\poppler-0.68.0\bin')
    #Saving pages in jpeg format
        for page in pages:
            page.save(path +"/"+'test1.png', 'PNG')
        os.remove(path + "/" + file)
    
for j,y in enumerate(myPiclist):
        
    img_need_aligned = cv2.imread(path +"/"+y)
    img_need_aligned = cv2.cvtColor(img_need_aligned, cv2.COLOR_BGR2RGB)

    im2Gray = cv2.cvtColor(img_need_aligned, cv2.COLOR_RGB2GRAY)   
    kp2, des2 = orb.detectAndCompute(im2Gray,None)

    # Matching feature
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)

    # Sort match by score
    matches.sort(key=lambda x: x.distance)

    # Draw top matches
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches( img_need_aligned, kp2, img_template, kp1, good, None, flags=2)
      
    # Extract location of good matches
    scrPoints = np.float32([kp2[match.queryIdx].pt for match in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[match.trainIdx].pt for match in good]).reshape(-1,1,2)

    # Find homography & wraping image
    M, _ = cv2.findHomography(scrPoints,dstPoints, cv2.RANSAC,5.0)
    imgScan = cv2.warpPerspective(img_need_aligned,M,(w,h))

#========== Extract data from image to text ==========
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
    myData = []

    print(f'################# Extracting Data from Form {j} ##########################')

    for x,r in enumerate(roi):

        cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
        imgCrop =imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        cv2.imwrite(str(x) +'.png', imgCrop)
        
        if r[2] =='text':
            # use VietOCR
            
            #==================processing image====================
            image = cv2.imread(str(x)+'.png')
            image = imgCrop.copy()
            or_image = pre_processing(image)
            img = Image.fromarray(cv2.cvtColor(or_image,cv2.COLOR_BGR2RGB))
           
            #======================================================
            
            img = (str(x)+'.png')
            img = Image.open(img)

            #================ xu ly so tien =======================
            if x==6:
                tien_so = str(detector.predict(img))
                number_change=int(re.sub("[^0-9]","",tien_so)) # Loai bo cac ky tu dac biet, chi lay so
                print(f'{r[3]} : {number_change}')
                myData.append(number_change)
            elif x==7:
                tien_chu_1 = str(detector.predict(img))
                tien_chu_1 = re.sub("[0-9]","",tien_chu_1)
                print(f'{r[3]} : {tien_chu_1}')
                myData.append(tien_chu_1)
            elif x==8:
                tien_chu = str(detector.predict(img))
                if len(tien_chu_1) == 0:
                    tien_chu = str(tien_chu)
                else:
                    tien_chu = str(tien_chu_1 + " " + tien_chu)

                tien_chu = re.sub("[0-9]","",tien_chu)
                text_change=int(w2n(tien_chu)) # Chuyen doi tu chu sang so
                if number_change > text_change:
                    number_change_text = n2w(str(number_change)) + str(" đồng") # Chuyen tu so sang chu
                    print(f'{r[3]} : {number_change_text}')
                    myData.append(number_change_text)
                else:
                    print(f'{r[3]} : {tien_chu}')
                    myData.append(tien_chu)
                
            else:
        
                print(f'{r[3]} : {detector.predict(img)}')

                myData.append(detector.predict(img))
     
        if r[2] == 'box':
            imgGray = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray, 127,255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            if totalPixels>pixelThreshold: totalPixels = 1
            else: totalPixels = 0
            print(f'{r[3]} : {totalPixels}')
            myData.append(totalPixels)

        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]), cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2) 
        
    # save data to file.csv
    myData = [e for e in myData if e !='']

    with open('dataoutput.csv','a+',encoding='utf-8') as f:
        f.write('\n')
        for data in myData:
            f.write((str(data)+str(',')))
    print(myData)    

cv2.waitKey(1)

