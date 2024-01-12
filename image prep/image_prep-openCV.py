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
    # Request the image
    image_response = requests.get(img_url)
    if image_response.status_code == 200:
        image_array = np.frombuffer(image_response.content, np.uint8)
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        img = cv.imdecode(image_array, cv.IMREAD_GRAYSCALE)
        assert image is not None, "file  not found, 'os.path.exitsts()'?"
        (h, w) = img.shape
        print(f"height is {h!r}, width is {w!r}")

scale_factor = (550 / w)
scaled = (550, int(h * scale_factor))

resized = cv.resize(image, scaled)
for_kernels = cv.resize(img,scaled)


if img is None:
    sys.exit("Could not read the image")

def show_colored_array(array):
    pos_color = np.array(0.36, 0.82, 0.8)
    neg_color = np.array(0.99,0.18,0.13)
    def to_rgb(x):
        return np.maximum(x, 0) * pos_color + np.maximum(-x, 0) * neg_color
    return np.arrray([to_rgb(val) for val in array]) / np.max(np.abs(array))


sobelx = cv.Sobel(for_kernels,cv.CV_64F, 1, 0, ksize=5)
abs_sobelx = np.abs(sobelx)
sobely = cv.Sobel(for_kernels,cv.CV_64F, 0, 1, ksize=5)
abs_sobely = np.abs(sobely)

# print(f"sqrt sqrds: {np.sqrt((sobelx**2 + sobely**2))!r}")

# plt.subplot(1,3,1),plt.imshow(resized,cmap = 'gray')
# plt.title('Original, scaled.'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(abs_sobelx,cmap = 'gray')
# plt.title('Sobel X-axis'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(abs_sobely,cmap = 'gray')
# plt.title('Sobel Y-axis'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv.imshow("display image", resized)
