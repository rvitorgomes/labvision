import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# kernels
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

def drawRectangle(original, contours):
	for contour in contours:
		[x, y, w, h] = cv2.boundingRect(contour)
		marked = cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 255), 2)
		return marked

def drawCircle(original, contours):
	for contour in contours:
		(x,y),radius = cv2.minEnclosingCircle(contour)
		if radius > 2 and radius < 7:
			center = (int(x),int(y))
			radius = int(radius)
			cv2.circle(original,center,radius,(0,255,0),2)
		# circled = cv2.drawContours(original, [contour], -1, (0, 255, 0), 2)
	return original


def main(argv):

	img = cv2.imread('lab_teste_y_fim.png', 0)

	# please kkk
	blur = cv2.GaussianBlur(img,(5,5),0)

	dilation = cv2.dilate(blur,kernel_ellipse,iterations = 2)
	cv2.imshow('dilation', dilation)

	erosion = cv2.erode(dilation,kernel_ellipse,iterations = 1)
	cv2.imshow('erosion', erosion)
	canny = cv2.Canny(erosion,45,70)
	cv2.imshow('canny', canny)

	(_,cnts,hierarchy) = cv2.findContours(canny.copy(), cv2.FILLED, cv2.CHAIN_APPROX_SIMPLE)

	draw = drawCircle(img, cnts)
	cv2.imshow('results', draw)
	cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv)



# compute the center of the contour
# M = cv2.moments(contour)
# cX = int(M["m10"] / M["m00"])
# cY = int(M["m01"] / M["m00"])
# cv2.circle(original, (cX, cY), 7, (255, 255, 255), -1)