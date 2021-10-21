import cv2
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import fnmatch
from numpy.lib.function_base import append
from re import match
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from pdf2image import convert_from_path
from vietnam_number import w2n, n2w

def main():
    
    if 'button_id' not in st.session_state:
        st.session_state['button_id'] = ''
    if 'color_to_label' not in st.session_state:
        st.session_state['color_to_label'] = {}
    PAGES = {"Project decription": about,
             "Extract text from files": from_file,
             "Extract text from webcam": from_cam,
             }

    page = st.sidebar.selectbox('Menu', options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown('<h6>Made by Vu Thanh Tung</h6>', unsafe_allow_html=True)

def about():
    max_width_str = f"max-width: 1000px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True,) 
    st.markdown( "<h1 style='text-align: center;'>Vietnamese Printed & Handwritten Text Extraction</h1>", unsafe_allow_html=True,)
    st.markdown( "<p style='text-align:center;'> Ho Chi Minh - 2021 - Vu Thanh Tung</p>", unsafe_allow_html=True) 
    st.image('Introduce/intro_1.png',use_column_width=True)
    st.markdown(
        """
    **Purpose:**
    - Extract printed and handwritten Vietnamese text from an existing form
    - Save the extracted data to a '.csv' file

    **Steps to work:**
    - Sample preparation
    - Processing and editing forms
    - Select the data area you want to extract
    - Model & Predict
    - Save data to Database
    """
    )
    st.markdown("[:pencil2: (Project decription detail)](https://hackmd.io/@tung1510/SkJYyGoHt)")

def from_file():
    max_width_str = f"max-width: 1000px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True,) 
    st.markdown( "<h1 style='text-align: center;'>Vietnamese Printed & Handwritten Text Extraction</h1>", unsafe_allow_html=True,)
    st.markdown( "<p style='text-align:center;'> Ho Chi Minh - 2021 - Vu Thanh Tung</p>", unsafe_allow_html=True) 
               
    file_uploaded = st.file_uploader("File upload", type = ['.jpg', '.png', '.jpeg', '.pdf'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        image.save("image.png", "PNG")
        
        myData = main_processing()
        myData = [e for e in myData if e !='']

        st.image('image.png',use_column_width=True)

        st.write(str(myData))
    # save data to file.csv

    with open('dataoutput.csv','a+',encoding='utf-8') as f:
        f.write('\n')
        for data in myData:
            f.write((str(data)+str(',')))

     # visualize my dataframe
    df = load_database()
    st.title("Data base") 
    st.write(df)

    cv2.waitKey(0)

def from_cam():
    path = "userforms"
    max_width_str = f"max-width: 1000px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True,) 
    st.markdown( "<h1 style='text-align: center;'>CAPTURE IMAGE FROM WEBCAM</h1>", unsafe_allow_html=True)
    st.markdown( "<p style='text-align:center;'> Ho Chi Minh - 2021 - Vu Thanh Tung</p>", unsafe_allow_html=True) 
    st.info("Webcam show on local computer ONLY! Press 's' to SAVE - Press 'q' to QUIT")

    camera = cv2.VideoCapture(0)
    # set DPI camera
    camera.set(3, 1280)
    camera.set(4, 720)
    
    if not camera.isOpened():
        raise IOError("Cannot open webcam")

    while (True):
        ret,image=camera.read()
        cv2.imshow('imshow',image)

        key = cv2.waitKey(1)&0xFF
        if key==ord('s'):
            cv2.imwrite("Image.png",image)
            break
        elif key==ord('q'):
            break

    myData = main_processing()
    myData = [e for e in myData if e !='']
    
    st.image('Image.png',use_column_width=True)
    st.write(str(myData))

    # save data to file.csv
    with open('dataoutput.csv','a+',encoding='utf-8') as f:
        f.write('\n')
        for data in myData:
            f.write((str(data)+str(',')))

    # visualize my dataframe
    df = load_database()
    st.title("Data base") 
    st.write(df)

    camera.release()
    cv2.destroyAllWindows()

#--------------- Function for processing ----------------

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


def load_database():
    df = pd.read_csv("dataoutput.csv")
    return df

def main_processing():
    myData=[]
    detector = Predictor(config)
    #========== Alignment Image for sample form ===========
    # Detect feature
    img_template = cv2.imread('Sample.png')
    h,w,c = img_template.shape
    im1Gray = cv2.cvtColor(img_template,cv2.COLOR_RGB2GRAY) 

    orb = cv2.ORB_create(max_feature)
    kp1, des1 = orb.detectAndCompute(im1Gray,None)
                    
    img_need_aligned = cv2.imread('Image.png')
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
    
    for x,r in enumerate(roi):

        cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
        imgCrop =imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        # cv2.imwrite(str(x) +'.png', imgCrop)
        
        if r[2] =='text':
            # use VietOCR
            img = (str(x)+'.png')
            img = Image.open(img)
     
            if x==6:
                tien_so = str(detector.predict(img))
                number_change=int(re.sub("[^0-9]",'',tien_so)) 
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
                    continue

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
                
        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]), cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2) 

    return myData
    
#-------------------------------------------------------
if __name__=="__main__":

    roi = [[(343, 180), (376, 205), 'text', 'ngay'], 
        [(419, 184), (448, 206), 'text', 'thang'], 
        [(483, 177), (565, 207), 'text', 'nam'], 
        [(230, 205), (752, 243), 'text', 'nguoi_nop'], 
        [(121, 243), (765, 271), 'text', 'dia_chi'], 
        [(145, 273), (751, 304), 'text', 'ly_do_nop'], 
        [(122, 305), (364, 335), 'text', 'tien_so'],    
        [(501, 305), (751, 335), 'text', 'tien_chu_1'], 
        [(57, 330), (751, 370), 'text', 'tien_chu_2']]

    per = 25
    pixelThreshold = 500
    max_feature = 5000
  
    st.set_page_config(page_title="Text Extraction", page_icon=":pencil2:")
    st.sidebar.subheader("Vietnamese Printed & Handwritten Text Extraction")

    #========== Config to use model vietocr ==========
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    config['predictor']['beamsearch']=False
   
    main()



