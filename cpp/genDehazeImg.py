# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dehaze1

windowSize = [15, 15]
ratio = 0.001
t0 = 0.1
parameter_w = 0.95


img = cv2.imread('./imageOfHePaper/image_He6.jpg')
darkChannelMat = dehaze1.darkChannelOfImage(img, windowSize)
atmosphericLight = dehaze1.atmosphericLightOfImg(darkChannelMat, img, ratio)
print 'atmosphericLight', atmosphericLight
# atmosphericLight = dehaze.atmosphericLightOfImg2(darkChannelMat, img, ratio)

transmission = dehaze1.transmissionOfImage(img, atmosphericLight, windowSize, parameter_w)
dehazedImg = dehaze1.recoverImage(img, atmosphericLight, t0, transmission)
dehazedImg = dehaze1.normlizedOfImg(dehazedImg)

cv2.imwrite('./dehazeOfHe_My/transmission_6dehaze1.jpg', transmission)
cv2.imwrite('./dehazeOfHe_My/darkChannel_6dehaze1.jpg', darkChannelMat)
cv2.imwrite('./dehazeOfHe_My/dehaze_6dehaze1.jpg', dehazedImg)