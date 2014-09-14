# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dehaze1
import dehaze
import dehaze3
import time
windowSize = [15, 15]
ratio = 0.001
t0 = 0.1
parameter_w = 0.95
img = cv2.imread('image_He9.png')


print "begin time of dehaze: ", time.localtime(time.time())
darkChannelMat = dehaze.darkChannelOfImage(img, windowSize)
atmosphericLight = dehaze.atmosphericLightOfImg(darkChannelMat, img, ratio)
transmission = dehaze.transmissionOfImage(img, atmosphericLight, windowSize, parameter_w)
dehazedImg = dehaze.recoverImage(img, atmosphericLight, t0, transmission)
dehazedImg = dehaze.normlizedOfImg(dehazedImg)
print "end time of dehaze: ", time.localtime(time.time())

cv2.imwrite('./dehazeOfHe_My/darkChannel_9dehaze.png', darkChannelMat)
cv2.imwrite('./dehazeOfHe_My/dehaze_9dehaze.png', dehazedImg)


print "begin time of dehaze1: ", time.localtime(time.time())
darkChannelMat1 = dehaze1.darkChannelOfImage(img, windowSize)
atmosphericLight1 = dehaze1.atmosphericLightOfImg(darkChannelMat1, img, ratio)
transmission1 = dehaze1.transmissionOfImage(img, atmosphericLight1, windowSize, parameter_w)
dehazedImg1 = dehaze1.recoverImage(img, atmosphericLight1, t0, transmission1)
dehazedImg1 = dehaze1.normlizedOfImg(dehazedImg1)
print "end time of dehaz1: ", time.localtime(time.time())

cv2.imwrite('./dehazeOfHe_My/darkChannel_9dehaze1.png', darkChannelMat1)
cv2.imwrite('./dehazeOfHe_My/dehaze_9dehaze1.png', dehazedImg1)

print "begin time of dehaze3: ", time.localtime(time.time())
darkChannelMat3 = dehaze3.darkChannelOfImage(img, windowSize)
atmosphericLight3 = dehaze3.atmosphericLightOfImg(darkChannelMat3, img, ratio)
transmission3 = dehaze3.transmissionOfImage(img, atmosphericLight3, windowSize, parameter_w)
dehazedImg3 = dehaze3.recoverImage(img, atmosphericLight3, t0, transmission3)
dehazedImg3 = dehaze3.normlizedOfImg(dehazedImg3)
print "end time of dehaze3: ", time.localtime(time.time())

cv2.imwrite('./dehazeOfHe_My/darkChannel_9dehaze3.png', darkChannelMat3)
cv2.imwrite('./dehazeOfHe_My/dehaze_9dehaze3.png', dehazedImg3)


h_dark, w_dark = darkChannelMat1.shape
h_t, w_t = transmission.shape

print "darkChannelMat shape", h_dark, w_dark
print "transmission shape", h_t, w_t

print darkChannelMat.shape
print darkChannelMat1.shape
print darkChannelMat3.shape
print transmission.shape
print transmission1.shape
print transmission3.shape

print atmosphericLight
print atmosphericLight1
print atmosphericLight3


n = 0; m = 0; p = 0; q = 0; k = 0; l = 0 
for i in range(h_t):
	for j in range(w_t):
		if transmission[i, j] != transmission1[i, j]:
			n += 1
		if darkChannelMat[i, j] != darkChannelMat1[i, j]:
			m += 1
		if transmission[i, j] != transmission3[i, j]:
			p += 1
		if darkChannelMat[i, j] != darkChannelMat3[i, j]:
			q += 1
		if transmission1[i, j] != transmission3[i, j]:
			k += 1
		if darkChannelMat1[i, j] != darkChannelMat3[i, j]:
			l += 1

print "sizeOfDrakChannel", h_dark*w_dark
print "sizeOfT", h_t*w_t
print "the num of dif dehaze and dehaze1 darkChannelMat:  ", n
print "the num of dif dehaze and dehaze1 T:  ", m
print "the num of dif dehaze and dehaze3 darkChannelMat:  ", p
print "the num of dif dehaze and dehaze3 T:  ", q
print "the num of dif dehaze1 and dehaze3 darkChannelMat:  ", k
print "the num of dif dehaze1 and dehaze3 T:  ", l



