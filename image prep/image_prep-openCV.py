import sys
from io import BytesIO

import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

import requests
from bs4 import BeautifulSoup

img = cv.imread('2401.jpg', cv.IMREAD_GRAYSCALE)
(h, w) = img.shape
assert img is not None, "file could not be read, check with os.path.exitsts()"


scale_factor = (550 / w)
scaled = (550, int(h * scale_factor))
resized = cv.resize(img, scaled)


if img is None:
    sys.exit("Could not read the image")

cv.imshow("display img", resized)

laplacian = cv.Laplacian(resized, cv.CV_64F)

#might try next
# scharr =

sobelx = cv.Sobel(resized,cv.CV_64F, 1, 0, ksize=5)
abs_sobelx = np.abs(sobelx)

sobely = cv.Sobel(resized,cv.CV_64F, 0, 1, ksize=5)
abs_sobely = np.abs(sobely)


# dtype output of uint8 (CV_8U) of |64F| is garbage
# sobel8ux = np.uint8(abs_sobelx)
# sobel8uy = np.uint8(abs_sobely)

plt.subplot(1,3,1),plt.imshow(resized,cmap = 'gray')
plt.title('Original, scaled.'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(abs_sobelx,cmap = 'gray')
plt.title('Sobel X-axis'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(abs_sobely,cmap = 'gray')
plt.title('Sobel Y-axis'), plt.xticks([]), plt.yticks([])
 

plt.show()

# k = cv.waitKey(0)

# if k == ord('s'):
#     cv.imwrite('deepField.jpg', img)