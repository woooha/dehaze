import cv2
import numpy as np

img = cv2.imread('./imageOfHePaper/image_He3.jpg')
img1 = cv2.imread('./imageOfDenseFog/5.jpg')
smallImg5 = img1[20:120, 80:180]
cv2.imwrite('smallImg5.jpg', smallImg5)

print img1.shape

smallImg = img[80: 180, 80 :180]

cv2.imwrite('smallImg.jpg',smallImg)
