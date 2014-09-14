# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dehaze1
import dehaze
import dehaze3
windowSize = [15, 15]
ratio = 0.001
t0 = 0.1
parameter_w = 0.95


img = cv2.imread('./imageOfHePaper/image_He9.png')
darkChannelMat = dehaze.darkChannelOfImage(img, windowSize)
atmosphericLight = dehaze.atmosphericLightOfImg(darkChannelMat, img, ratio)

transmission = dehaze.transmissionOfImage(img, atmosphericLight, windowSize, parameter_w)
dehazedImg = dehaze.recoverImage(img, atmosphericLight, t0, transmission)
dehazedImg = dehaze.normlizedOfImg(dehazedImg)

cv2.imwrite('./dehazeOfHe_My/darkChannel_9dehaze1.png', darkChannelMat)
cv2.imwrite('./dehazeOfHe_My/dehaze_9dehaze1.png', dehazedImg)




darkChannelMat1 = dehaze1.darkChannelOfImage(img, windowSize)
atmosphericLight1 = dehaze1.atmosphericLightOfImg(darkChannelMat1, img, ratio)

transmission1 = dehaze1.transmissionOfImage(img, atmosphericLight1, windowSize, parameter_w)
dehazedImg1 = dehaze1.recoverImage(img, atmosphericLight1, t0, transmission1)
dehazedImg1 = dehaze1.normlizedOfImg(dehazedImg1)

cv2.imwrite('./dehazeOfHe_My/darkChannel_9dehaze2.png', darkChannelMat1)
cv2.imwrite('./dehazeOfHe_My/dehaze_9dehaze2.png', dehazedImg1)




darkChannelMat3 = dehaze3.darkChannelOfImage(img, windowSize)
atmosphericLight3 = dehaze3.atmosphericLightOfImg(darkChannelMat3, img, ratio)

transmission3 = dehaze3.transmissionOfImage(img, atmosphericLight3, windowSize, parameter_w)
dehazedImg3 = dehaze3.recoverImage(img, atmosphericLight3, t0, transmission3)
dehazedImg3 = dehaze3.normlizedOfImg(dehazedImg3)

cv2.imwrite('./dehazeOfHe_My/darkChannel_9dehaze2.png', darkChannelMat3)
cv2.imwrite('./dehazeOfHe_My/dehaze_9dehaze2.png', dehazedImg3)

print darkChannelMat.shape
h1, w1 = darkChannelMat1.shape
h, w = transmission.shape
print atmosphericLight
print atmosphericLight1
print atmosphericLight2

n = 0
m = 0
p = 0
q = 0
for i in range(h):
	for j in range(w):
		if transmission[i, j] != transmission1[i, j]:
			n += 1
		if darkChannelMat[i, j] != darkChannelMat1[i, j]:
			m += 1
		if transmission[i, j] != transmission2[i, j]:
			p += 1
		if darkChannelMat[i, j] != darkChannelMat2[i, j]:
			q += 1

print h*w
print h1*w1
print n
print m
print p
print q


