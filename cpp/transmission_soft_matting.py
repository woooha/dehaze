# -*- coding: utf8 -*- 
import cv2
import numpy as np
import dehaze
from scipy.sparse import lil_matrix

# Estimating the transmission after soft matting 
def transmission_after_matting(img, transmission, L, parameter_lambda):
	row, col = L.shape
	h, w = img.shape[: 2]
	eye_matrix = np.eye((row))
	transmission = transmission.reshape(1, row)
	print "adafd", L.shape
	print transmission.shape
	transmission_matting = parameter_lambda * transmission * ((L + parameter_lambda * eye_matrix).I)
	transmission_matting = transmission_matting.reshape(h, w)
	return transmission_matting

# 求Laplacian矩阵
def Laplacian_matrix(img, epsilon, windowSize_matting):
	h, w = img.shape[: 2]
	L = lil_matrix((h * w, h * w))
	for row in range(h*w):
	# 求L的元素值
		for col in range(row+1, h*w):
			if row == col:
				delta = 1
			else:
				delta = 0
			value = elementOfL(img, row, col, epsilon, delta, windowSize_matting)
			if value != 0:
				L[row, col] = value
				L[col, row] = L[row, col]
	return L

def elementOfL(img, row, col, epsilon, delta, windowSize_matting):
	h, w = img.shape[: 2]
	numPixels = windowSize_matting^2
	eye_matrix = np.eye((3))
	# 选择的点i和点j，在原图像中的位置
	firstPixelRowOfImg = row / w
	firstPixelColOfImg = row % w
	secondPixelRowOfImg = col / w
	secondPixelColOfImg = col % w
	# 选择满足条件的窗口，选择可以包括点i和点j的窗口
	value = 0 
	if abs((firstPixelRowOfImg - secondPixelRowOfImg)) < windowSize_matting and abs((firstPixelColOfImg - secondPixelColOfImg)) < windowSize_matting:
		# 找到窗口移动的坐标
		rowOfBegin = max(firstPixelRowOfImg, secondPixelRowOfImg) - windowSize_matting
		colOfBegin = max(firstPixelColOfImg, secondPixelColOfImg) - windowSize_matting
		rowOfEnd = min(firstPixelRowOfImg, secondPixelRowOfImg)
		colOfEnd = min(firstPixelColOfImg, secondPixelColOfImg)
		# 判断边缘
		if rowOfBegin < 0:
			rowOfBegin = 0
		if colOfBegin < 0:
			colOfBegin = 0
		if rowOfEnd + windowSize_matting > h:
			rowOfEnd = h
		if colOfEnd + windowSize_matting > w:
			colOfEnd = w
		# the mean and covariance matrix of the colors in windows 
		for i in range(rowOfBegin, rowOfEnd):
			for j in range(colOfBegin, colOfEnd):
				windowOfImg = img[i : i+windowSize_matting, j : j+windowSize_matting, :]
				size1, size2 = windowOfImg.shape[: 2]
				num = size1*size2
				windowOfMat = [windowOfImg[:,:,0].reshape(1, num)[0], windowOfImg[:,:,1].reshape(1,num)[0], windowOfImg[:,:,2].reshape(1,num)[0]]
				meanOfWindow = np.mean(windowOfMat, axis=1)
				covarianceOfWindow = np.cov(windowOfMat)
				value += delta - ((1 + np.matrix(img[firstPixelRowOfImg, firstPixelColOfImg] - meanOfWindow)) \
					* np.matrix((covarianceOfWindow + (epsilon / numPixels) * eye_matrix)).I \
					* np.matrix(img[secondPixelRowOfImg, secondPixelColOfImg] - meanOfWindow).T) / numPixels
	return value

# windowSize = [15, 15]
# ratio = 0.001
# t0 = 0.1
# parameter_w = 0.95
# epsilon = 0.0001
# parameter_lambda = 1e-3
# windowSize_matting = 3

# # img = cv2.imread('smallImg.jpg')
# img = cv2.imread('./smallImg.jpg')
# darkChannelMat = dehaze.darkChannelOfImage(img, windowSize)
# atmosphericLight = dehaze.atmosphericLightOfImg(darkChannelMat, img, ratio)
# transmission = dehaze.transmissionOfImage(img, atmosphericLight, windowSize, parameter_w)

# L = Laplacian_matrix(img, epsilon, windowSize_matting)
# transmission_matting = transmission_after_matting(transmission, L, parameter_lambda)
# # print "transmission_matting", transmission_matting

# cv2.imshow('transmission', transmission)
# cv2.imshow('transmission_matting', transmission_matting)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


