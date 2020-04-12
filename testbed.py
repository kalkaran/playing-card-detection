import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pickle
from glob import glob
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
import Augmentor


"""

Backgrounds file:
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar xf dtd-r1.0.1.tar.gz to dtd folder.


"""


"""
Various test images paths:

    #img=cv2.imread("./test/all_mads.jpg")
    #img=cv2.imread("./test/depositphotos_184840322-stock-photo-single-spades-playing-card-gamble.jpg")
    #img=cv2.imread("./test/96066795-single-of-spades-playing-card-for-gamble-playing-cards-2-isolated-on-black-background-great-for-any-.jpg")
    #img=cv2.imread("./test/CW_Cards_Africanqueen.jpg")
    squashpath = "./test/squash2.jpg"
    squash = cv2.imread(squashpath)
    
"""
data_dir="data" # Directory that will contain all kinds of data (the data we download and the data we generate)

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

card_suits=['s','h','d','c']
card_values=['A','K','Q','J','10','9','8','7','6','5','4','3','2']

# Pickle file containing the background images from the DTD
backgrounds_pck_fn=data_dir+"/backgrounds.pck"

# Pickle file containing the card images
cards_pck_fn=data_dir+"/cards.pck"


# imgW,imgH: dimensions of the generated dataset images
imgW=720
imgH=720

cv2_resource_path="./venv/lib/python3.7/site-packages/cv2/data/"



def picture():
    img=cv2.imread("./test/green_screen.png")
    cv2.imshow("output",img)
    cv2.waitKey(0)



def video():
    cap = cv2.VideoCapture("test/2c.avi")
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    while True:
        success, img = cap.read()
        cv2.imshow("Video",img)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break


def imgconversiongray():
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    cv2.imshow("output",imgGray)
    cv2.waitKey(0)



def imgconversiongauss():
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
    cv2.imshow("output",imgBlur)
    cv2.waitKey(0)

def imgconversionCanny():
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
    imgCanny = cv2.Canny(img, 100,100)
    cv2.imshow("output", imgCanny)
    cv2.waitKey(0)
    imgCanny2 = cv2.Canny(imgGray, 100,100)
    cv2.imshow("output", imgCanny2)
    cv2.waitKey(0)
    imgCanny3 = cv2.Canny(imgBlur, 100, 200)
    cv2.imshow("output", imgCanny3)
    cv2.waitKey(0)


def imgconversionDialation():
    kernel = np.ones((5,5),np.uint8)
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
    imgCanny = cv2.Canny(img, 100,100)
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=5)
    cv2.imshow("output", imgDialation)
    cv2.waitKey(0)

def imgErosion():
    kernel = np.ones((5,5),np.uint8)
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
    imgCanny = cv2.Canny(img, 100,100)
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=5)
    cv2.imshow("output", imgDialation)
    cv2.waitKey(0)
    imgErosi = cv2.dilate(imgDialation, kernel, iterations=5)
    cv2.imshow("output", imgErosi)
    cv2.waitKey(0)

def imageResize():
    img = cv2.imread("./test/green_screen.png")
    print(img.shape)
    imgResize = cv2.resize(img,(300,200))
    print(imgResize.shape)
    cv2.imshow("output", imgResize)
    cv2.waitKey(0)


def imageCrop():
    img = cv2.imread("./test/green_screen.png")
    print(img.shape)
    # - y, then - x.
    imgCrop = img[100:200, 1000:1200]
    cv2.imshow("output", imgCrop)
    cv2.waitKey(0)



def createImg():
    #np dimensions, then type = np.uint8 = 0-255
    img = np.zeros((512,512,3),np.uint8)
    print(img.shape)

    img[100:200, 200:250] = 255, 0, 0
    img[200:300, 200:250] = 0, 255, 0
    img[300:400, 200:250] = 0, 0, 255

    # img x,y start, x,y finish
    # cv2.line(img, (0, 0), (img.shape[1], 500), (0, 0, 255), (5))

    cv2.rectangle(img, (1, 1), (250, 350), (0, 0, 255), 2, cv2.FILLED)

    cv2.circle(img,(250, 350), 30, (0, 0, 255))
    cv2.putText(img, " PUFFIN ", (300, 300), cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0), 1)

    cv2.imshow("output", img)
    cv2.waitKey(0)


def warpImage():
    img = cv2.imread("./test/squash2.jpg")
    img2 = cv2.imread("./test/green_screen.png")
    print(img.shape)

    cornerA = [1350, 1720]
    cornerB = [2290, 1850]
    cornerC = [440, 2580]
    cornerD = [1885, 2980]
    cv2.circle(img, (1350, 1720), 20, (255, 0, 255), 5)
    cv2.circle(img, (2290, 1850), 20, (0, 0, 255), 5)
    cv2.circle(img, (440, 2580), 20, (0, 255, 255), 5)
    cv2.circle(img, (1885, 2980), 20, (255, 0, 0), 5)
    pts1 = (np.float32([cornerA, cornerB, cornerC, cornerD]))
    #new size
    width = 57*10
    height = 87*10
    pts2 = (np.float32([[0, 0], [width, 0], [0, height], [width, height]]))
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imshow("output2",img2)
    cv2.imshow("outputpic", img)
    cv2.imshow("output", imgOutput)
    cv2.waitKey(0)

def stackimage(img):
    imgHor = np.hstack((img, img))
    imgVer = np.vstack((img, img))
    cv2.imshow(imgHor)
    cv2.imshow(imgVer)
    cv2.waitKey(0)


def empty(arg):
    pass

def imageHSV():
    path = "./test/squash2.jpg"
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 1200, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

    while True:
        img = cv2.imread(path)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)

        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        #find out what this does - mask
        mask = cv2.inRange(imgHSV, lower, upper)

        cv2.imshow("Original", img)
        cv2.imshow("HSV", imgHSV)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def shape_recognition():
    pathsq = "./test/squash2.jpg"
    imgsq = cv2.imread(pathsq)
    img=cv2.imread("./test/all_mads.jpg")
    img = cv2.imread("./dataset2_blackbackground/2c.jpg")
    #img=cv2.imread("./data/cards/2c-min.jpg")
    #img=cv2.imread("./data/cards/sshot2.png")
    #img=cv2.imread("/Users/n/PycharmProjects/playing-card-detection/data/cards/2c-sizemin.jpg")
    #img=cv2.imread("./test/depositphotos_184840322-stock-photo-single-spades-playing-card-gamble.jpg")
    #img=cv2.imread("./test/96066795-single-of-spades-playing-card-for-gamble-playing-cards-2-isolated-on-black-background-great-for-any-.jpg")
    #img=cv2.imread("./test/CW_Cards_Africanqueen.jpg")
    #img = cv2.imread("./data/cards/2c-min.jpg")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    #imgCanny = cv2.Canny(imgGray, 0, 197)
    imgCanny = cv2.Canny(imgBlur, 0, 197)
    imgBlank = np.zeros_like(img)
    imgContours = img.copy()

    getContours(imgCanny, imgContours, imgBlank)

    imgsqGray = cv2.cvtColor(imgsq, cv2.COLOR_BGRA2GRAY)
    imgsqBlur = cv2.GaussianBlur(imgsqGray, (7, 7), 1)
    imgsqCanny = cv2.Canny(imgsqBlur, 20, 10)
    imgsqBlank = np.zeros_like(imgsq)
    imgsqContours = imgsq.copy()

    #getContours(imgsqCanny, imgsqContours,imgsqBlank)

    imgStack = stackImages(.7, [img, imgBlur, imgCanny, imgContours, imgBlank])
    #imgStack = stackImages(.7, [img, imgCanny, imgBlank])
    #imgsqStack = stackImages(0.2, [imgsq, imgsqBlur, imgsqCanny, imgsqContours, imgsqBlank])



    cv2.imshow("Stack", imgStack)
    #cv2.imshow("Stacksq", imgsqStack)
    cv2.waitKey(0)


def getContours(img, imgContour, imgBlank):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 10)
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            #looking at connected shapes - this is the TRUE
            approx = cv2.approxPolyDP(cnt, 0.02*peri,True)
            #len here gives how many connected edges.
            objCor = (len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            objectType = ""
            cardAspectRatio = 57/87
            devation = 0.10
            if objCor == 4:
                aspectRatio = w/float(h)
                if aspectRatio > (1 - devation) * cardAspectRatio and aspectRatio < (1 + devation) * cardAspectRatio:
                    objectType = "Card"
            cv2.rectangle(imgBlank,(x,y),(x+w, y+h), (0,255,0),2)
            cv2.putText(imgBlank, objectType,
                        (x + (w // 2) - 20, y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

#NB might need to create custom cascades.



def face_recognition():
    """NB. file has been removed."""
    face_cascade = cv2.CascadeClassifier(cv2_resource_path+"haarcascade_frontalface_default.xml")
    img = cv2.imread("IMG_4703.JPG")

    img_gray = cv2.cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, 1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)


    cv2.imshow("Result", img)
    cv2.waitKey(0)





def getContours_dataset2(img, imgContour):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 10)
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            #looking at connected shapes - this is the TRUE
            approx = cv2.approxPolyDP(cnt, 0.02*peri,True)
            #len here gives how many connected edges.
            objCor = (len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            objectType = ""
            cardAspectRatio = 57/87
            devation = 0.10
            if objCor == 4:
                aspectRatio = w/float(h)
                if aspectRatio > (1 - devation) * cardAspectRatio and aspectRatio < (1 + devation) * cardAspectRatio:
                    objectType = "Card"
            cv2.rectangle(img,(x,y),(x+w, y+h), (0,255,0),2)
            cv2.putText(img, objectType,
                        (x + (w // 2) - 20, y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)



def card_prep():
    img1 = cv2.imread("./dataset2_blackbackground/2c.jpg")
    img2 = cv2.imread("./dataset2_blackbackground/2h.jpg")
    img3 = cv2.imread("./dataset2_blackbackground/Ad.jpg")
    imgStack = stackImages(0.4,[img1,img2,img3])

    def callback(foo):
        pass

    # create windows and trackbar
    cv2.namedWindow('parameters')
    cv2.createTrackbar('threshold1', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('threshold2', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('apertureSize', 'parameters', 0, 2, callback)
    cv2.createTrackbar('L1/L2', 'parameters', 0, 1, callback)

    while (True):
        # get threshold value from trackbar
        th1 = cv2.getTrackbarPos('threshold1', 'parameters')
        th2 = cv2.getTrackbarPos('threshold2', 'parameters')

        # aperture size can only be 3,5, or 7
        apSize = cv2.getTrackbarPos('apertureSize', 'parameters') * 2 + 3

        # true or false for the norm flag
        norm_flag = cv2.getTrackbarPos('L1/L2', 'parameters') == 1

        # print out the values
        print('')
        print('threshold1: {}'.format(th1))
        print('threshold2: {}'.format(th2))
        print('apertureSize: {}'.format(apSize))
        print('L2gradient: {}'.format(norm_flag))

        cv2.imshow("orginal", imgStack)
        gray = cv2.cvtColor(imgStack, cv2.COLOR_BGRA2GRAY)
        # Convert in gray color
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Noise-reducing and edge-preserving filter
        gray=cv2.bilateralFilter(gray,5,75,75)
        # Edge extraction
        #edge=cv2.Canny(gray, 0, 197, apertureSize=3)
        imgGray = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        imgCanny = cv2.Canny(imgGray, 0, 197)
        imgCanny = cv2.Canny(gray, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
        #imgCanny = cv2.Canny(imgBlur, 0, 197)
        imgBlank = np.zeros_like(img1)
        imgContours = img1.copy()

        getContours(imgCanny, imgContours, imgBlank)
        #getContours_dataset2(gray, edge)
        imgStack = stackImages(.4, [img1, imgBlur, imgCanny, imgContours, imgBlank])
        #edge = cv2.Canny(gray, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
        cv2.imshow('stack', imgStack)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()




def card_prep2():
    img1 = cv2.imread("./dataset2_blackbackground/2c.jpg")
    img2 = cv2.imread("./dataset2_blackbackground/2h.jpg")
    img3 = cv2.imread("./dataset2_blackbackground/Ad.jpg")
    imgStack = stackImages(0.4, [img1, img2, img3])
    cv2.imshow("orginal", imgStack)


    def callback(foo):
        pass

    # create windows and trackbar
    cv2.namedWindow('parameters')
    cv2.createTrackbar('threshold1', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('threshold2', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('apertureSize', 'parameters', 0, 2, callback)
    cv2.createTrackbar('L1/L2', 'parameters', 0, 1, callback)
    while (True):
        # get threshold value from trackbar
        th1 = cv2.getTrackbarPos('threshold1', 'parameters')
        th2 = cv2.getTrackbarPos('threshold2', 'parameters')

        # aperture size can only be 3,5, or 7
        apSize = cv2.getTrackbarPos('apertureSize', 'parameters') * 2 + 3

        # true or false for the norm flag
        norm_flag = cv2.getTrackbarPos('L1/L2', 'parameters') == 1

        # print out the values
        print('')
        print('threshold1: {}'.format(th1))
        print('threshold2: {}'.format(th2))
        print('apertureSize: {}'.format(apSize))
        print('L2gradient: {}'.format(norm_flag))

        cv2.imshow("orginal", imgStack)
        #gray = cv2.cvtColor(imgStack, cv2.COLOR_BGRA2GRAY)
        #gray = cv2.bilateralFilter(gray, 5, 75, 75)
        imgGray = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        #imgCanny = cv2.Canny(imgGray, 0, 197, apertureSize=3)
        imgCanny = cv2.Canny(imgBlur, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
        imgBlank = np.zeros_like(img1)
        imgContours = img1.copy()

        getContours(imgCanny, imgContours, imgBlank)

        cv2.imshow('Contours', imgContours)
        cv2.imshow('imgBlank', imgBlank)
        #imgStack1 = stackImages(.7, [img1, imgBlur, imgCanny, imgContours, imgBlank])
        #cv2.imshow('stack', imgStack1)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break


def card_extract(img, output_fn=None):
    #card settings:
    # nealcards
    cardW = 56
    cardH = 86
    cornerXmin = 2
    cornerXmax = 8
    cornerYmin = 4
    cornerYmax = 21

    # We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
    # You shouldn't need to change this
    zoom = 4
    cardW *= zoom
    cardH *= zoom
    cornerXmin = int(cornerXmin * zoom)
    cornerXmax = int(cornerXmax * zoom)
    cornerYmin = int(cornerYmin * zoom)
    cornerYmax = int(cornerYmax * zoom)


    #misc variables from jupyter
    # imgW,imgH: dimensions of the generated dataset images
    imgW = 720
    imgH = 720

    refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
    refCardRot = np.array([[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32)
    refCornerHL = np.array(
        [[cornerXmin, cornerYmin], [cornerXmax, cornerYmin], [cornerXmax, cornerYmax], [cornerXmin, cornerYmax]],
        dtype=np.float32)
    refCornerLR = np.array([[cardW - cornerXmax, cardH - cornerYmax], [cardW - cornerXmin, cardH - cornerYmax],
                            [cardW - cornerXmin, cardH - cornerYmin], [cardW - cornerXmax, cardH - cornerYmin]],
                           dtype=np.float32)
    refCorners = np.array([refCornerHL, refCornerLR])

    #alphamask:
    bord_size = 2  # bord_size alpha=0
    alphamask = np.ones((cardH, cardW), dtype=np.uint8) * 255
    cv2.rectangle(alphamask, (0, 0), (cardW - 1, cardH - 1), 0, bord_size)
    cv2.line(alphamask, (bord_size * 3, 0), (0, bord_size * 3), 0, bord_size)
    cv2.line(alphamask, (cardW - bord_size * 3, 0), (cardW, bord_size * 3), 0, bord_size)
    cv2.line(alphamask, (0, cardH - bord_size * 3), (bord_size * 3, cardH), 0, bord_size)
    cv2.line(alphamask, (cardW - bord_size * 3, cardH), (cardW, cardH - bord_size * 3), 0, bord_size)
    plt.figure(figsize=(10, 10))
    plt.imshow(alphamask)



    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 0, 197, apertureSize=3)
    contours, hierachy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)

    areaCnt=cv2.contourArea(cnt)
    areaBox=cv2.contourArea(box)

    ((xr, yr), (wr, hr), thetar) = rect

    if wr > hr:
        Mp = cv2.getPerspectiveTransform(np.float32(box), refCard)
    else:
        Mp = cv2.getPerspectiveTransform(np.float32(box), refCardRot)

    imgwarp = cv2.warpPerspective(img, Mp, (cardW, cardH))
    # Add alpha layer
    imgwarp = cv2.cvtColor(imgwarp, cv2.COLOR_BGR2BGRA)
    # Shape of 'cnt' is (n,1,2), type=int with n = number of points
    # We reshape into (1,n,2), type=float32, before feeding to perspectiveTransform
    cnta = cnt.reshape(1, -1, 2).astype(np.float32)
    # Apply the transformation 'Mp' to the contour
    cntwarp = cv2.perspectiveTransform(cnta, Mp)
    cntwarp = cntwarp.astype(np.int)

    # We build the alpha channel so that we have transparency on the
    # external border of the card
    # First, initialize alpha channel fully transparent
    alphachannel = np.zeros(imgwarp.shape[:2], dtype=np.uint8)
    # Then fill in the contour to make opaque this zone of the card
    cv2.drawContours(alphachannel, cntwarp, 0, 255, -1)

    # Apply the alphamask onto the alpha channel to clean it
    alphachannel = cv2.bitwise_and(alphachannel, alphamask)

    # Add the alphachannel to the warped image
    imgwarp[:, :, 3] = alphachannel

    # Save the image to file
    if output_fn is not None:
        cv2.imwrite(output_fn, imgwarp)

    return imgwarp


# img1 = cv2.imread("./dataset2_blackbackground/2c.jpg")
#img2 = cv2.imread("./dataset2_blackbackground/2h.jpg")
# img3 = cv2.imread("./dataset2_blackbackground/Ad.jpg")
#
# cardexample = card_extract(img1)
# cv2.imshow('Contours', cardexample)
# cv2.waitKey(0)


def extract_all():
    card_suits = ['s', 'h', 'd', 'c']
    card_values = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
    dir = "dataset2_blackbackground/"
    extension = "jpg"
    imgs_dir = "data/cards"
    for suit in card_suits:
        for value in card_values:

            card_name = value + suit
            print("extracting" + card_name)
            file = os.path.join(dir, card_name + "." + extension)
            output_dir = os.path.join(imgs_dir, card_name)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            output_file = output_dir + "/" + card_name + ".jpg"
            img = cv2.imread(file)
            card_extract(img, output_file)



# img2=cv2.imread("data/cards/10s/10s.jpg")
# cv2.imshow("test",img2)
# cv2.waitKey(0)





def augment_images():
    print("aug")
    imgs_dir = "data/cards"
    imgs_fns = glob(imgs_dir + "/*")
    print(imgs_fns)
    for img in imgs_fns:
        print(img)
        augment_function(img)

def augment_function(img):
    print(type(img))
    p = Augmentor.Pipeline(img)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=1, min_factor=0.8, max_factor=1.5)
    p.random_brightness(probability=0.7, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=0.7, min_factor=0.8, max_factor=1.2)
    p.skew_top_bottom(probability=0.7, magnitude=0.1)
    p.sample(1)
    print(type(img))

#augment_images()

#find hulls:
#this function is taken from someone else
#adding some variables to from the jupyter notebook to fix function

data_dir="data" # Directory that will contain all kinds of data (the data we download and the data we generate)

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

card_suits=['s','h','d','c']
card_values=['A','K','Q','J','10','9','8','7','6','5','4','3','2']

# Pickle file containing the background images from the DTD
backgrounds_pck_fn=data_dir+"/backgrounds.pck"

# Pickle file containing the card images
cards_pck_fn=data_dir+"/cards.pck"


# imgW,imgH: dimensions of the generated dataset images
imgW=720
imgH=720



#nealcards
""" NB. the corners on our card set is not consistent. so I will choose the most inclusive area.
further the measurements asked seem wrong Ymax should be inclusive of Ymin."""
cardW=56
cardH=86
cornerXmin=2
cornerXmax=8
cornerYmin=4
cornerYmax=21

refCard=np.array([[0,0],[cardW,0],[cardW,cardH],[0,cardH]],dtype=np.float32)

refCardRot=np.array([[cardW,0],[cardW,cardH],[0,cardH],[0,0]],dtype=np.float32)

refCornerHL=np.array([[cornerXmin,cornerYmin],
                      [cornerXmax,cornerYmin],
                      [cornerXmax,cornerYmax],
                      [cornerXmin,cornerYmax]],dtype=np.float32)

refCornerLR=np.array([[cardW-cornerXmax,cardH-cornerYmax],
                      [cardW-cornerXmin,cardH-cornerYmax],
                      [cardW-cornerXmin,cardH-cornerYmin],
                      [cardW-cornerXmax,cardH-cornerYmin]],dtype=np.float32)

refCorners=np.array([refCornerHL,refCornerLR])



"""
This is a test function to find the value part of the card
looks like the the corner it is finding is way off.
will have to code my own solution
will start with upper left.

further it looks like the corner explanation image in the original is wrong.

"""
def findHull_imageAnalysis(img, corner):
    kernel = np.ones((3, 3), np.uint8)
    corner = corner.astype(np.int)
    print(img.shape)

    y = img.shape[0]
    x = img.shape[1]

    cardW = 56
    cardH = 86

    factor_y = int(y / cardH)
    factor_x = int(x / cardW)


    cornerXmin = 3
    cornerXmax = 9
    cornerYmin = 4
    cornerYmax = 21

    #coordinates
    x1 = int(factor_x * cornerXmin)
    x2 = int(factor_x * cornerXmax)
    y1 = int(factor_y * cornerYmin)
    y2 = int(factor_y * cornerYmax)

    cornerA = [x1, y1]
    cornerB = [x1, y2]
    cornerC = [x2, y1]
    cornerD = [x2, y2]

    #top left corner.
    """ NB. the corners on our card set is not consistent. so I will choose the most inclusive area."""
    print("x1 " + str(x1))
    print("x2 " + str(x2))
    print("y2 " + str(y1))
    print("y1 " + str(y2))


    print(cornerA)
    print(cornerB)
    print(cornerC)
    print(cornerD)

    # We will focus on the zone of 'img' delimited by 'corner'
    # x1 = int(corner[0][0])
    # y1 = int(corner[0][1])
    # x2 = int(corner[2][0])
    # y2 = int(corner[2][1])
    # print("x1 " + str(x1))
    # print("x2 " + str(x2))
    # print("y1 " + str(y1))
    # print("y1 " + str(y2))
    w = x2 - x1
    h = y2 - y1
    zone = img[y1:y2,x1:x2]
    print(zone.shape)

    return zone

imghull = cv2.imread("./data/cards/Kh/Kh.jpg")
print(type(imghull))
# cv2.imshow("preimg",imghull)
# cv2.waitKey(0)
#cv2.imshow("I", findHull_imageAnalysis(imghull, refCornerLR))
cv2.imwrite("./test/croptest.jpg",findHull_imageAnalysis(imghull, refCornerLR))






def findHull(img, corner=refCornerHL):
    """
        this function is taken from Jupyternotebook.
        Find in the zone 'corner' of image 'img' and return, the convex hull delimiting
        the value and suit symbols
        'corner' (shape (4,2)) is an array of 4 points delimiting a rectangular zone,
        takes one of the 2 possible values : refCornerHL or refCornerLR
    """

    kernel = np.ones((3, 3), np.uint8)
    corner = corner.astype(np.int)

    # We will focus on the zone of 'img' delimited by 'corner'
    x1 = int(corner[0][0])
    y1 = int(corner[0][1])
    x2 = int(corner[2][0])
    y2 = int(corner[2][1])
    w = x2 - x1
    h = y2 - y1
    zone = img[y1:y2, x1:x2].copy()


    strange_cnt = np.zeros_like(zone)
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    thld = cv2.Canny(gray, 30, 200)
    thld = cv2.dilate(thld, kernel, iterations=1)

    cv2.imshow("handled", thld)
    cv2.waitKey(0)

    # Find the contours
    contours, _ = cv2.findContours(thld.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 30  # We will reject contours with small area. TWEAK, 'zoom' dependant
    min_solidity = 0.3  # Reject contours with a low solidity. TWEAK

    concat_contour = None  # We will aggregate in 'concat_contour' the contours that we want to keep

    ok = True
    for c in contours:
        area = cv2.contourArea(c)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        # Determine the center of gravity (cx,cy) of the contour
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #  abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 : TWEAK, the idea here is to keep only the contours which are closed to the center of the zone
        if area >= min_area and abs(w / 2 - cx) < w * 0.3 and abs(h / 2 - cy) < h * 0.4 and solidity > min_solidity:
            # if debug != "no":
            #     cv2.drawContours(zone, [c], 0, (255, 0, 0), -1)
            if concat_contour is None:
                concat_contour = c
            else:
                concat_contour = np.concatenate((concat_contour, c))
        # if debug != "no" and solidity <= min_solidity:
        #     print("Solidity", solidity)
        #     cv2.drawContours(strange_cnt, [c], 0, 255, 2)
        #     cv2.imshow("Strange contours", strange_cnt)

    if concat_contour is not None:
        # At this point, we suppose that 'concat_contour' contains only the contours corresponding the value and suit symbols
        # We can now determine the hull
        hull = cv2.convexHull(concat_contour)
        hull_area = cv2.contourArea(hull)
        # If the area of the hull is to small or too big, there may be a problem
        min_hull_area = 940  # TWEAK, deck and 'zoom' dependant
        max_hull_area = 2120  # TWEAK, deck and 'zoom' dependant
        if hull_area < min_hull_area or hull_area > max_hull_area:
            ok = False
            # if debug != "no":
            #     print("Hull area=", hull_area, "too large or too small")
        # So far, the coordinates of the hull are relative to 'zone'
        # We need the coordinates relative to the image -> 'hull_in_img'
        hull_in_img = hull + corner[0]

    else:
        ok = False

    # if debug != "no":
    #     if concat_contour is not None:
    #         cv2.drawContours(zone, [hull], 0, (0, 255, 0), 1)
    #         cv2.drawContours(img, [hull_in_img], 0, (0, 255, 0), 1)
    #     cv2.imshow("Zone", zone)
    #     cv2.imshow("Image", img)
    #     if ok and debug != "pause_always":
    #         key = cv2.waitKey(1)
    #     else:
    #         key = cv2.waitKey(0)
    #     if key == 27:
    #         return None
    # if ok == False:
    #     return None

    return hull_in_img



def main():
    #to extra cards from dataset
    #extract_all()
    #to find hulls. - requires cards to be extracted.
    imghull = cv2.imread("./data/cards/2c/2c.jpg")
    #findHull(imghull)





main()






