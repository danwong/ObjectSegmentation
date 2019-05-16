import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

def main():

    img = cv2.imread('tree1.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    cv2.imwrite("tree1-gray.jpg", gray)



    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    cv2.imwrite("tree1-sure.jpg", sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    """
    cv2.imshow('image', sure_fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """


    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    cv2.imwrite("sure_fg.jpg", sure_fg)
    cv2.imwrite("unkown.jpg", unknown)

    nonzero = cv2.countNonZero(sure_bg)
    total = sure_bg.shape[0] * sure_bg.shape[1]
    zero = total - nonzero
    ratio = zero * 100 / float(total)
    print 'ratio of zero / nonzero = {:.2f}'.format(ratio)


    #find canny edges
    edged = cv2.Canny(sure_bg, 30, 200)
    """
    cv2.imshow('canny edges', edged)
    cv2.waitKey(0)
    """
    cv2.imwrite("canny_edges.jpg", edged)


    contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)



    #use -1 as the 3rd parameter to draw all the contours
    cv2.drawContours(img,contours,-1,(0,255,0),3)
    cv2.imwrite("contoured_img.jpg", img)
    """
    cv2.imshow('contours',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
