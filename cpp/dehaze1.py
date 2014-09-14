# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 暗通道：min(c)、min(每个像素点)
# 透射率t：min(c)、min(每个像素点)

# 计算暗通道图
def darkChannelOfImage(img, windowSize):
	h, w = img.shape[: 2]
	gray = np.zeros((h, w))
	darkChannelMat = np.zeros((h, w))
	minValue = 255

	# 计算一张图片的最小channel值，得到只有一个通道的灰度图
	for row in range(h):
		for col in range(w):
			gray[row, col] = np.min(img[row, col, :])
	
	# 计算图片的每个点，求dark  value 
	for row in range(h):
		for col in range(w):
			darkChannelMat[row, col] = darkValue(gray, row, col, windowSize, minValue)
	return darkChannelMat

# 计算一个点的Dark Channel
def darkValue(grayImg, row, col, windowSize, minValue):
	h, w = grayImg.shape[: 2]
	windowRowBegin = row - windowSize[0] / 2
	
	if windowRowBegin < 0: 
		windowRowBegin = 0
	windowRowEnd = windowRowBegin + int(windowSize[0])
	if windowRowEnd > h: 
		windowRowEnd = h

	windowColBegin = col - windowSize[1] / 2
	if windowColBegin < 0: 
		windowColBegin = 0
	windowColEnd = windowColBegin + windowSize[1]
	if windowColEnd > w: 
		windowColEnd = w
	for i in range(windowRowBegin, windowRowEnd):
		for j in range(windowColBegin, windowColEnd):
			if grayImg[i, j] < minValue:
				minValue = grayImg[i, j]
	return minValue

# Estimation the Atmospheric Light，用brightestPixel函数在原图中找到的像素点，求平均，作为A的值
# A是3维的，对应彩色图的三个通道
def atmosphericLightOfImg(darkChannelMat, originalImage, ratio):
	pixels, numSelectPixels = pixelOfDarkChannelImage(darkChannelMat, ratio)
	maxAtmosphericLight = 220
	atmosphericLight = np.zeros((1, 3))
	for pixel in pixels:
		brightPixel = brightestPixel(originalImage, pixel)
		atmosphericLight += brightPixel	
	atmosphericLight = atmosphericLight / numSelectPixels
	atmosphericLight = atmosphericLight[0]
	for i in range(3):
		if atmosphericLight[i] > maxAtmosphericLight:
			atmosphericLight[i] = maxAtmosphericLight
	return atmosphericLight

# Estimation the Atmospheric Light, brightestPixel函数在原图中找到像素点，A取最亮的点
def atmosphericLightOfImg2(darkChannelMat, originalImage, ratio):
	pixels, numSelectPixels = pixelOfDarkChannelImage(darkChannelMat, ratio)
	atmosphericLight = np.zeros((1, 3))
	brightest = 0
	for pixel in pixels:
		brightPixel = brightestPixel(originalImage, pixel)
		brightestOfPixel = sum(brightPixel[0])
		if brightestOfPixel > brightest:
			brightest = brightestOfPixel
			atmosphericLight = brightPixel[0]	
	return atmosphericLight

# 暗通道图中按照亮度的大小取前0.1%的像素
def pixelOfDarkChannelImage(img, ratio):
	h, w = img.shape[: 2]
	numPixels = h * w 
	numSelectPixels = int(numPixels * ratio)
	newImg = np.zeros(h*w)
	pixels = []
	k = 0
	newImg = img.reshape(1, h*w)[0]
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
	gray = np.zeros((h, w))
	imgDiviA = np.zeros((originalImage.shape))
	minValue = np.Inf
	# 计算原图中每个像素点除以atmosphericLight，
	imgDiviA[:, :, 0] = originalImage[:, :, 0] / float(atmosphericLight[0])
	imgDiviA[:, :, 1] = originalImage[:, :, 1] / float(atmosphericLight[1])
	imgDiviA[:, :, 2] = originalImage[:, :, 2] / float(atmosphericLight[2])

	# 单通道图，每个像素点取三个颜色的最小值
	for row in range(h):
		for col in range(w):
			gray[row, col] = np.min(imgDiviA[row, col, :])

	for row in range(h):
		for col in range(w):
			minValue = darkValue(gray, row, col, windowSize, minValue)
			transmission[row, col] = 1 - parameter_w * minValue
	return transmission

# Recovering the Scene Radiance
def recoverImage(img, atmosphericLight, t0, transmission):
	h, w, channel = img.shape
	dehazedImg = np.zeros((h, w, channel))
	a = np.zeros((h, w, channel))
	b = np.zeros((h, w, channel))

	for row in range(h):
		for col in range(w):
			dehazedImg[row, col] = ((img[row, col] - atmosphericLight) / float(max(t0, transmission[row, col]))) + atmosphericLight
	return dehazedImg

# 对像素点的值进行归一化
def normlizedOfImg(img):
	h, w, channel = np.shape(img)
	maxPixel = img.max()
	minPixel = img.min()
	normlizedImg = 255 * ((img - minPixel) / (maxPixel - minPixel))
	for row in range(h):
		for col in range(w):
			for channel in range(channel):
				normlizedImg[row, col, channel] = int(normlizedImg[row, col, channel])
	return normlizedImg

# 对像素点的值进行归一化
# def normlizedOfImg1(img):
# 	h, w, channel = np.shape(img)
# 	maxPixel = img.max()
# 	minPixel = img.min()
# 	normlizedImgList = []
# 	normlizedImg = 255 * ((img - minPixel) / (maxPixel - minPixel))
# 	normlizedImgList.extend(normlizedImg[:,:,0].reshape(1,h*w)[0])
# 	normlizedImgList.extend(normlizedImg[:,:,1].reshape(1,h*w)[0])
# 	normlizedImgList.extend(normlizedImg[:,:,2].reshape(1,h*w)[0])
# 	normlizedImg = map(lambda x:int(x), normlizedImgList)
# 	normlizedImg = np.array(normlizedImg)

# 	normlizedImg = normlizedImg.reshape(h, w, channel)
# 	return normlizedImg

def normlizedOfImg2(img):
	h, w= img.shape
	maxPixel = img.max()
	minPixel = img.min()
	normlizedImg = 255 * ((img - minPixel) / (maxPixel - minPixel))
	normlizedImg = normlizedImg.reshape(1, h*w)[0]
	normlizedImg = map(lambda x:int(x), normlizedImg)
	normlizedImg = normlizedImg.reshape(h, w)
	return normlizedImg







		

