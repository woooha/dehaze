import cv2
import numpy as np

img = cv2.imread('/Users/bushanshan/Documents/Workspace/haze_removal/imageOfHePaper/image_He9.png')

smallImg5 = img[0:100, 105:205]
cv2.imwrite('smallImgHe9.png', smallImg5)


