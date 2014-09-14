# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 暗通道：min(每个像素点)、min(c)
# 透射率t：min(每个像素点)、min(c)、

# 计算暗通道图
def darkChannelOfImage(img, windowSize):
	h, w = img.shape[: 2]
	darkChannelMat = np.zeros((h, w))

	for row in range(h):
		for col in range(w):
			minValue = darkValue(img, row, col, windowSize)
			darkChannelMat[row, col] = min(minValue)
	return darkChannelMat

# 计算一个点的Dark Channel
def darkValue(img, row, col, windowSize):
	h, w = img.shape[: 2]
	minValue = [255, 255, 255]
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
			if img[i, j, 0] < minValue[0]: minValue[0] = img[i, j, 0]
			if img[i, j, 1] < minValue[1]: minValue[1] = img[i, j, 1]
			if img[i, j, 2] < minValue[2]: minValue[2] = img[i, j, 2]
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
	imgDiviAtmoLight = np.zeros((originalImage.shape))
	# 计算原图中每个像素点除以atmosphericLight，
	imgDiviAtmoLight[:, :, 0] = originalImage[:, :, 0] / float(atmosphericLight[0])
	imgDiviAtmoLight[:, :, 1] = originalImage[:, :, 1] / float(atmosphericLight[1])
	imgDiviAtmoLight[:, :, 2] = originalImage[:, :, 2] / float(atmosphericLight[2])

	for row in range(h):
		for col in range(w):
			minValue = minOfImgDiviAtmoLight(imgDiviAtmoLight, row, col, windowSize)
			transmission[row, col] = 1 - parameter_w * min(minValue)
	return transmission

# 分别计算每个颜色通道在一个窗口内的最小值
def minOfImgDiviAtmoLight(imgDiviAtmoLight, row, col, windowSize):
	h, w = imgDiviAtmoLight.shape[: 2]
	minValue = [np.Inf, np.Inf, np.Inf]
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
			if imgDiviAtmoLight[i, j, 0] < minValue[0]: minValue[0] = imgDiviAtmoLight[i, j, 0]
			if imgDiviAtmoLight[i, j, 1] < minValue[1]: minValue[1] = imgDiviAtmoLight[i, j, 1]
			if imgDiviAtmoLight[i, j, 2] < minValue[2]: minValue[2] = imgDiviAtmoLight[i, j, 2]
	return minValue

# Recovering the Scene Radiance
def recoverImage(img, atmosphericLight, t0, transmission):
	h, w, channel = img.shape
	dehazedImg = np.zeros((h, w, channel))
	a = np.zeros((h, w, channel))
	b = np.zeros((h, w, channel))

	for row in range(h):
		for col in range(w):
		# 	a[row, col] = (img[row, col] - atmosphericLight)
		# 	b[row, col] = a[row, col] / float(max(t0, transmission[row, col]))
		# 	dehazedImg[row, col] = b[row, col] + atmosphericLight
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

def normlizedOfImg2(img):
	h, w= img.shape
	maxPixel = img.max()
	minPixel = img.min()
	normlizedImg = 255 * ((img - minPixel) / (maxPixel - minPixel))
	for row in range(h):
		for col in range(w):
			normlizedImg[row, col] = int(normlizedImg[row, col])
	return normlizedImg

windowSize = [15, 15]
ratio = 0.001
t0 = 0.1
parameter_w = 0.95


# img = cv2.imread('./imageOfHePaper/image_He6.jpg')
# darkChannelMat = darkChannelOfImage(img, windowSize)
# atmosphericLight = atmosphericLightOfImg(darkChannelMat, img, ratio)
# # print 'atmosphericLight', atmosphericLight
# # atmosphericLight = dehaze.atmosphericLightOfImg2(darkChannelMat, img, ratio)

# transmission = transmissionOfImage(img, atmosphericLight, windowSize, parameter_w)
# dehazedImg = recoverImage(img, atmosphericLight, t0, transmission)
# dehazedImg = normlizedOfImg(dehazedImg)

# cv2.imwrite('./dehazeOfHe_My/transmission_6dehaze3.jpg', transmission)
# cv2.imwrite('./dehazeOfHe_My/darkChannel_6dehaze3.jpg', darkChannelMat)
# cv2.imwrite('./dehazeOfHe_My/dehaze_6dehaze3.jpg', dehazedImg)





		

