# -*- coding: utf-8 -*-
import cv2
import numpy as np


# 计算暗通道图
def darkChannelOfImage(img, windowSize):
	h, w = img.shape[: 2]
	gray = np.zeros((h, w), dtype='uint8')
	darkChannelMat = np.zeros((h, w), dtype='uint8')

	# 计算一张图片的最小channel值，得到只有一个通道的灰度图
	for row in range(h):
		for col in range(w):
			gray[row, col] = np.min(img[row, col, :])
	#计算图片的每个点，求dark  value 
	for row in range(h):
		for col in range(w):
			darkChannelMat[row, col] = darkValue(gray, row, col, windowSize)
	return darkChannelMat


# 计算一个点的Dark Channel
def darkValue(img, row, col, windowSize):
	h, w = img.shape[: 2]
	minValue = 255
	windowRowBegin = row - windowSize[0] / 2
	if windowRowBegin < 0: windowRowBegin = 0
	windowRowEnd = windowRowBegin + int(windowSize[0])
	if windowRowEnd > h: windowRowEnd = h

	windowColBegin = col - windowSize[1] / 2
	if windowColBegin < 0: windowColBegin = 0
	windowColEnd = windowColBegin + windowSize[1]
	if windowColEnd > w: windowColEnd = w
	for i in range(windowRowBegin, windowRowEnd):
		for j in range(windowColBegin, windowColEnd):
			if img[i, j] < minValue:
				minValue = img[i, j]
	return minValue

# Estimation the Atmospheric Light，用brightestPixel函数在原图中找到的像素点，求平均，作为A的值
# A是3维的，对应彩色图的三个通道
def atmosphericLightOfImg(darkChannelMat, originalImage, ratio):
	pixels, numSelectPixels = pixelOfDarkChannelImage(darkChannelMat, ratio)
	atmosphericLight = np.zeros((1, 3))
	for pixel in pixels:
		brightPixel = brightestPixel(originalImage, pixel)
		atmosphericLight += brightPixel	
	atmosphericLight = atmosphericLight / numSelectPixels
	atmosphericLight = atmosphericLight[0]
	return atmosphericLight

# 暗通道图中按照亮度的大小取前0.1%的像素
def pixelOfDarkChannelImage(img, ratio):
	h, w = img.shape[: 2]
	numPixels = h * w 
	numSelectPixels = int(numPixels * ratio)
	newImg = np.zeros(h*w, dtype='uint8')
	pixels = []
	k = 0
	for row in range(h):
		newImg[k*w : (k+1)*w] = img[row, :]
		k += 1
	pixelIndexs = cv2.sortIdx(newImg, cv2.SORT_EVERY_COLUMN + cv2.SORT_DESCENDING)
	for i in range(numSelectPixels):
		rowOfPixel = pixelIndexs[i] / w
		colOfPixel = pixelIndexs[i] % w
		pixels.append([rowOfPixel, colOfPixel])
	return pixels, numSelectPixels


#在原始有雾图像中寻找与pixelOfDarkChannelImage对应的像素点的值
def brightestPixel(img, pixel):
	brightPixel = img[pixel[0], pixel[1]]
	return brightPixel

# Estimation the Transmission，一个和图片大小相同的矩阵
def transmissionOfImage(originalImage, atmosphericLight, windowSize, parameter_w):
	h, w = originalImage.shape[: 2]
	transmission = np.zeros((h, w))
	imgDiviAtmoLight = np.zeros((img.shape))
	imgDiviAtmoLight[:, :, 0] = img[:, :, 0] / float(atmosphericLight[0])
	imgDiviAtmoLight[:, :, 1] = img[:, :, 1] / float(atmosphericLight[1])
	imgDiviAtmoLight[:, :, 2] = img[:, :, 2] / float(atmosphericLight[2])

	for row in range(h):
		for col in range(w):
			minValue = minOfImgDiviAtmoLight(imgDiviAtmoLight, row, col, windowSize)
			print min(minValue)
			transmission[row, col] = 1 - parameter_w * min(minValue)
	return transmission

# 计算原图中每个像素点除以atmosphericLight，
def minOfImgDiviAtmoLight(imgDiviAtmoLight, row, col, windowSize):
	h, w = img.shape[: 2]
	minValue = [np.Inf, np.Inf, np.Inf]
	windowRowBegin = row - windowSize[0] / 2
	if windowRowBegin < 0: windowRowBegin = 0
	windowRowEnd = windowRowBegin + int(windowSize[0])
	if windowRowEnd > h: windowRowEnd = h
	windowColBegin = col - windowSize[1] / 2
	if windowColBegin < 0: windowColBegin = 0
	windowColEnd = windowColBegin + windowSize[1]
	if windowColEnd > w: windowColEnd = w

	for i in range(windowRowBegin, windowRowEnd):
		for j in range(windowColBegin, windowColEnd):
			if imgDiviAtmoLight[i, j, 0] < minValue[0]: minValue[0] = imgDiviAtmoLight[i, j, 0]
			if imgDiviAtmoLight[i, j, 1] < minValue[1]: minValue[1] = imgDiviAtmoLight[i, j, 1]
			if imgDiviAtmoLight[i, j, 2] < minValue[2]: minValue[2] = imgDiviAtmoLight[i, j, 2]
	return minValue

# Recovering the Scene Radiance
def recoverImage(img, atmosphericLight, t0, transmission):
	h, w, channel = img.shape
	dehazedImg = np.zeros((h, w, channel))
	for row in range(h):
		for col in range(w):
			dehazedImg[row, col] = ((img[row, col] - atmosphericLight) / float(max(t0, transmission[row, col]))) + atmosphericLight
	return dehazedImg




img = cv2.imread('8.jpg')
windowSize = [15, 15]
ratio = 0.001
t0 = 0.1
parameter_w = 0.95
print img

darkChannelMat = darkChannelOfImage(img, windowSize)
print darkChannelMat
atmosphericLight = atmosphericLightOfImg(darkChannelMat, img, ratio)
print 'atmosphericLight', atmosphericLight
transmission = transmissionOfImage(img, atmosphericLight, windowSize, parameter_w)
print transmission

dehazedImg = recoverImage(img, atmosphericLight, t0, transmission)
print 'dadafd',dehazedImg[0]
print 'adadfa', dehazedImg[1]
print 'sadfad', dehazedImg[2]
cv2.imshow('img', img)
cv2.imshow('darkChannelMat', darkChannelMat)
cv2.imshow('dehazedImg', dehazedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


		

