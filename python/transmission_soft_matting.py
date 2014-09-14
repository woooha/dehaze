# -*- coding: utf8 -*- 
import cv2
import numpy as np
import dehaze
from scipy import sparse
from scipy.sparse import linalg
import time

# Estimating the transmission after soft matting 
def transmission_after_matting(img, transmission, L, parameter_lambda):
	row, col = L.shape
	h, w = img.shape[: 2]
	eye_matrix = np.eye((row))
	# eye_matrix = sparse.eye((row))
	transmission = transmission.reshape(1, row)
	print "adafd", L.shape
	transmission_matting = parameter_lambda *transmission * (L + parameter_lambda * eye_matrix).I
	print "jkkjk"
	print time.localtime(time.time())
	print h, w 
	print transmission_matting.shape
	transmission_matting = transmission_matting.reshape(100, 100)
	return transmission_matting

# 求Laplacian矩阵
def Laplacian_matrix(img, epsilon, wind_matting):
	h, w = img.shape[: 2]
	L = sparse.lil_matrix((h * w, h * w))
	delta0 = 0
	delta1 = 1
	sizeOfWind = wind_matting[0] * wind_matting[1]
	# L对角线上元素值
	for i in range(h*w):
		L[i, i] = elementOfL(img, i, i, h, w, epsilon, delta1, wind_matting,sizeOfWind)

	for row in range(h*w):
	# 求L的非对角相元素值
		for col in range(row+1, h*w):
			value = elementOfL(img, row, col, h, w, epsilon, delta0, wind_matting,sizeOfWind)
			if value != 0:
				L[row, col] = value
				L[col, row] = L[row, col]
	return L

def elementOfL(img, row, col, h, w, epsilon, delta, wind_matting,sizeOfWind):
	eye_matrix = np.eye((3))
	# 选择的点i和点j，在原图像中的位置
	firstPixelRowOfImg = row / w
	firstPixelColOfImg = row % w
	secondPixelRowOfImg = col / w
	secondPixelColOfImg = col % w
	# 选择满足条件的窗口，选择可以包括点i和点j的窗口
	value = 0 
	if abs((firstPixelRowOfImg - secondPixelRowOfImg)) < wind_matting[0] and abs((firstPixelColOfImg - secondPixelColOfImg)) < wind_matting[1]:
		# 找到窗口移动的坐标
		rowOfBegin = max(firstPixelRowOfImg, secondPixelRowOfImg) - wind_matting[0]
		colOfBegin = max(firstPixelColOfImg, secondPixelColOfImg) - wind_matting[1]
		rowOfEnd = min(firstPixelRowOfImg, secondPixelRowOfImg)
		colOfEnd = min(firstPixelColOfImg, secondPixelColOfImg)
		# 判断边缘
		if rowOfBegin < 0:
			rowOfBegin = 0
		if colOfBegin < 0:
			colOfBegin = 0
		if rowOfEnd + wind_matting[0] > h:
			rowOfEnd = h - wind_matting[0]
		if colOfEnd + wind_matting[1] > w:
			colOfEnd = w - wind_matting[1]

		# the mean and covariance matrix of the colors in windows 
		for i in range(rowOfBegin, rowOfEnd):
			for j in range(colOfBegin, colOfEnd):
				windowOfImg = img[i : i+wind_matting[0], j : j+wind_matting[1], :]
				windowOfMat = [windowOfImg[:,:,0].reshape(1,sizeOfWind)[0], windowOfImg[:,:,1].reshape(1,sizeOfWind)[0], windowOfImg[:,:,2].reshape(1,sizeOfWind)[0]]
				meanOfWindow = np.mean(windowOfMat, axis=1)
				covarianceOfWindow = np.cov(windowOfMat)
				value += delta - ((1 + np.matrix(img[firstPixelRowOfImg, firstPixelColOfImg] - meanOfWindow)) \
					* np.matrix((covarianceOfWindow + (epsilon /sizeOfWind) * eye_matrix)).I \
					* np.matrix(img[secondPixelRowOfImg, secondPixelColOfImg] - meanOfWindow).T) /sizeOfWind
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


