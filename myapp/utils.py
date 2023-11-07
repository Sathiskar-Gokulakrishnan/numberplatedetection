import cv2
import numpy as np
import pytesseract
import logging
from ultralytics import YOLO
from django.template import loader
logger = logging.getLogger(__name__)

model = YOLO('myapp/runs/detect/train2/weights/best.pt')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image ,5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5 ,5) ,np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)


# erosion
def erode(image):
    kernel = np.ones((5 ,5) ,np.uint8)
    return cv2.erode(image, kernel, iterations = 1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5 ,5) ,np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

functions = [get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew]

def numberPlateToText(numberPlate):
    print("inside numberPlateToText")
    objectGray = get_grayscale(numberPlate)
    ret,objectThresh = cv2.threshold(objectGray,127,255,cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(objectThresh)
    print(f"OCR RESULT ===> {text}")
    #return matchByPattern(objectThresh)
    return text

def getNumberPlate(image):
    logger.info("getNumberPlate")
    print("getNumberPlate")
    try:
        predictions = model(image)
        cropedImage = image
        for prediction in predictions:
            boxes = prediction.boxes
            boxes = boxes.numpy()
            print(boxes)

            [xmin,ymin,xmax,ymax] = boxes.xyxy.tolist()[0]  # print the box coordinates
            print(xmin,ymin,xmax,ymax)
            xmin,ymin,xmax,ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            print(type(image))
            cropedImage = image[ymin:ymax, xmin:xmax]
        return cropedImage

    except Exception as err:
        logger.error(f"error while process image : {err}")
        return None
