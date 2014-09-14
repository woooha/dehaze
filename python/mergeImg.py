# -*- coding: utf-8 -*-

import numpy as np
import cv2

# 合并三张大小相同的图片
def mergeImageOfRow(img1, img2, img3, s):
	h, w, channel = img1.shape
 	merge = np.zeros((h, 3*w, channel), 'uint8')
 	merge[0 : h, 0 : w] = img1
 	merge[0 : h, w : 2*w] = img2
 	merge[0 : h, 2*w : 3*w] = img3
 	cv2.imwrite('./mergeImg/merge%s.bmp' %s, merge)
 	return merge

def mergeImageOfCol(img1, img2, img3, s):
	h, w, channel = img1.shape
 	merge = np.zeros((3*h, w, channel), 'uint8')
 	merge[0 : h, 0 : w] = img1
 	merge[h : 2*h, 0 : w] = img2
 	merge[2*h : 3*h, 0 : w] = img3
 	cv2.imwrite('./mergeImg/merge%s.bmp' %s, merge)
 	return merge

s = '10'
originalImg = cv2.imread('./denseFogImg/%s.bmp' %s)
dehazeImg = cv2.imread('./denseFOgResult/dehaze_%sw_0.8.bmp' %s)
dehazeImg_pre = cv2.imread('./denseFogResult_pre/preResult_%s.bmp' %s)


h, w, channel = originalImg.shape
if h > w:
	merge = mergeImageOfRow(originalImg, dehazeImg, dehazeImg_pre, s)
else:
	merge = mergeImageOfCol(originalImg, dehazeImg, dehazeImg_pre, s)