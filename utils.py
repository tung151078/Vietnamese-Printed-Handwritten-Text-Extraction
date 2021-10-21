import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from re import match
from PIL import Image


# ===== functions to processing image====

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
