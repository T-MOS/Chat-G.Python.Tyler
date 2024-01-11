import sys
from io import BytesIO

import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

from PIL import Image

import requests
from bs4 import BeautifulSoup

# Image retrieval
    # URL of APOD website
url = 'https://apod.nasa.gov/apod/astropix.html'
    # Send GET request to APOD website and parse HTML response with BeautifulSoup
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
        # Find image tag and get source URL
    img_tag = soup.find('img')
    img_url = 'https://apod.nasa.gov/apod/' + img_tag['src']
    # Request the image and open it with PIL
    image_response = requests.get(img_url)
    if image_response.status_code == 200:
        # image_buffer = BytesIO()
        image_array = np.frombuffer(image_response.content, np.uint8)
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        img = cv.imdecode(image_array, cv.IMREAD_GRAYSCALE)
        assert image is not None, "file  not found, 'os.path.exitsts()'?"
        (h, w) = img.shape
        print("height is {h}, width is {w}")

scale_factor = (550 / w)
scaled = (550, int(h * scale_factor))

resized = cv.resize(image, scaled)
for_kernels = cv.resize(img,scaled)


if img is None:
    sys.exit("Could not read the image")

cv.imshow("display img", resized)

#laplacian = cv.Laplacian(resized, cv.CV_64F)
# sobel_scharrx = cv.Sobel(resized, cv.CV_64F, 1, 0, ksize=-1)
# sobel_scharry = cv.Sobel(resized, cv.CV_64F, 0, 1, ksize=-1)
# abs_sobel_scharrx = np.abs(sobel_scharrx)
# abs_sobel_scharry = np.abs(sobel_scharry)


sobelx = cv.Sobel(for_kernels,cv.CV_64F, 1, 0, ksize=5)
abs_sobelx = np.abs(sobelx)

sobely = cv.Sobel(for_kernels,cv.CV_64F, 0, 1, ksize=5)
abs_sobely = np.abs(sobely)


plt.subplot(1,3,1),plt.imshow(resized,cmap = 'gray')
plt.title('Original, scaled.'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian 2nd Derivative'), plt.xticks([]), plt.yticks([])

plt.subplot(1,3,2),plt.imshow(abs_sobelx,cmap = 'gray')
plt.title('Sobel X-axis'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(abs_sobely,cmap = 'gray')
plt.title('Sobel Y-axis'), plt.xticks([]), plt.yticks([])

# plt.subplot(1,3,2),plt.imshow(abs_sobel_scharrx,cmap = 'gray')
# plt.title('Sobel X-axis'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(abs_sobel_scharry,cmap = 'gray')
# plt.title('Sobel Y-axis'), plt.xticks([]), plt.yticks([])


plt.show()

# k = cv.waitKey(0)

# if k == ord('s'):
#     cv.imwrite('deepField.jpg', img)
