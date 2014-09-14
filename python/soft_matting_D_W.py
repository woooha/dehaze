# -*- coding: utf8 -*- 
import cv2
import numpy as np
import dehaze
from scipy.sparse import csc_matrix

# Estimating the transmission after soft matting 
def transmission_after_matting(transmission, L, parameter_lambda):
	row, col = L.shape
	eye_matrix = np.eye((row))
	transmission_matting = parameter_lambda * transmission * ((L + parameter_lambda * eye_matrix).I)
	return transmission_matting

# 求Laplacian矩阵
def Laplacian_matrix(img, epsilon, windowSize_matting):
	h, w = img.shape[: 2]
	# W对称矩阵
	W = csc_matrix((h * w, h * w))
	# W = np.zeros((h*w, h*w))
	print "Size of L is  %d * %d" % (h * w, h * w)
	for row in range(h*w):
		if row % 1000 == 0:
			print row
		for col in range(row+1, h*w):
			value = elementOfW(img, row, col, epsilon, windowSize_matting)
			if value != 0:
			 	W[row, col] = value
				W[col, row] = W[row, col]
	# D对角矩阵
	print 'dadf',W
	diagonal = np.sum(W, axis = 0)
	D = diagonal * np.eye((h*w)) 
	L = D - W
	return L

# 计算W矩阵， L = D - W
def elementOfW(img, row, col, epsilon, windowSize_matting):
	h, w = img.shape[: 2]
	# The numberof pixels in the window w
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
				# print "adfd", np.matrix(img[secondPixelRowOfImg, secondPixelColOfImg] - meanOfWindow).shape
				# print np.matrix(img[firstPixelRowOfImg, firstPixelColOfImg]- meanOfWindow).shape
				# print "dsad", meanOfWindow
				# print "adfdf", np.matrix((covarianceOfWindow + (epsilon / numPixels) * eye_matrix)).I.shape
				value += ((1 + np.matrix(img[firstPixelRowOfImg, firstPixelColOfImg] - meanOfWindow)) \
					* np.matrix((covarianceOfWindow + (epsilon / numPixels) * eye_matrix)).I \
					* np.matrix(img[secondPixelRowOfImg, secondPixelColOfImg] - meanOfWindow).T) / numPixels
	return value