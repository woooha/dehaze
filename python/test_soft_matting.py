# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dehaze
import transmission_soft_matting
import time

windowSize = [15, 15]
ratio = 0.001
t0 = 0.1
parameter_w = 0.95

epsilon = 0.0001
parameter_lambda = 1e-3
windowSize_matting = [3, 3]

print "Begin:  ", time.localtime(time.time())
img = cv2.imread('smallImgHe9.png')
darkChannelMat = dehaze.darkChannelOfImage(img, windowSize)
atmosphericLight = dehaze.atmosphericLightOfImg(darkChannelMat, img, ratio)
transmission = dehaze.transmissionOfImage(img, atmosphericLight, windowSize, parameter_w)

L = transmission_soft_matting.Laplacian_matrix(img, epsilon, windowSize_matting)
print "the time of L:  ", time.localtime(time.time())
transmission_matting = transmission_soft_matting.transmission_after_matting(img, transmission, L, parameter_lambda)
dehazedImg_matting = dehaze.recoverImage(img, atmosphericLight, t0, transmission_matting)
dehazedImg_matting = dehaze.normlizedOfImg(dehazedImg_matting)


cv2.imwrite('darkChannel_softHe9.png', darkChannelMat)
cv2.imwrite('dehazedImg_softHe9.png', dehazedImg_matting)


