# -*- coding: utf-8 -*-
import cv2
import numpy as np

def darkChannlOfImage(img, windowSize):

	h, w = img.shape[: 2]
	gray = np.zeros((h, w), dtype='uint8')
	darkChannlOfImage = np.zeros((h, w), dtype='uint8')

	for row in range(h):
		for col in range(w):
			gray[row, col] = np.min(img[row, col, :])

	for row in range(h):
		for col in range(w):
			darkChannlOfImage[row, col] = darkValue(gray, row, col, windowSize)

	cv2.imshow('gray', gray)
	cv2.imshow('darkChannlOfImagea', darkChannlOfImagea)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def darkValue(img, row, col, windowSize):
	h, w = img.shape[: 2]
	minValue = 255
	windowRowBegin = row - windowSize[0] / 2
	if windowRowBegin < 0: windowRowBegin = 0
	windowRowEnd = windowRowBegin + int(windowSize[0])
	if windowRowEnd > w: windowRowEnd = w

	windowColBegin = col - windowSize[1] / 2
	if windowColBegin < 0: windowColBegin = 0
	windowColEnd = windowColBegin + windowSize[1]
	if windowColEnd > h: windowColEnd = h

	for i in range(windowRowEnd + 1):
		for j in range(windowColEnd + 1):
			if img[i, j] < minValue:
				minValue = img[i, j]
			# print minValue
	return minValue

img = cv2.imread('1.jpg')
windowSize = [16, 16]
darkChannlOfImage(img, windowSize)



		

