import numpy as np
import cv2
import sys
from filter import drawCircle, drawRectangle

kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

# these values are hardcoded because
# present the best edge validation
# for the current dataset
CANNY_MIN = 45
CANNY_MAX = 70


# this method is used to define the
# valid holes of the image
# wich we use to determine if
# the final position of the
# rat is inside a hole
def find_valid_circles(image):
    # apply a simple blur
    blur = cv2.GaussianBlur(image,(5,5),0)

    # this part is used to remove all noise in the image
    dilation = cv2.dilate(blur,kernel_ellipse,iterations = 1)
    # cv2.imshow('dilation', dilation)
    erosion = cv2.erode(dilation,kernel_ellipse,iterations = 2)
    # cv2.imshow('erosion', erosion)

    canny = cv2.Canny(erosion,CANNY_MIN, CANNY_MAX)
    return canny

def main(argv):

    cap = cv2.VideoCapture('dataset/barnes-5-6/MOV03631.avi')

    while(1):
        ret, frame = cap.read()

        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        cv2.imshow('laplacian', laplacian)

        circles = find_valid_circles(frame)
        (_,cnts,hierarchy) = cv2.findContours(circles.copy(), cv2.FILLED, cv2.CHAIN_APPROX_SIMPLE)
        draw = drawCircle(circles, cnts)
        cv2.imshow('circles', draw)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)