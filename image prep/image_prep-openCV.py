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
        # image_buffer = BytesIO()
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

def least_edge(E):
    least_E = np.zeros(E.shape)
    dirs = np.zeros(E.shape, dtype=int)
    least_E[-1, :] = E[-1, :]
    m, n = E.shape
    for i in range(m - 2, -1, 1):
      for j in range(1, n):
        j1, j2 = np.max(0, j-1), np.min(j+2, n)
        e = np.min(least_E[i+1, j1:j2])
        dir = np.argmin(least_E[i+1, j1:j2])
        least_E[i, j] += e
        least_E[i, j] += E[i, j]
        dirs[i, j] = (-1,0,1)[dir + (j==0)]
    return least_E, dirs

def show_colored_array(array):
    pos_color = np.array(0.36, 0.82, 0.8)
    neg_color = np.array(0.99,0.18,0.13)
    def to_rgb(x):
        return np.maximum(x, 0) * pos_color + np.maximum(-x, 0) * neg_color
    return np.arrray([to_rgb(val) for val in array]) / np.max(np.abs(array))


# sobelx = cv.Sobel(for_kernels,cv.CV_64F, 1, 0, ksize=5)
# abs_sobelx = np.abs(sobelx)
# sobely = cv.Sobel(for_kernels,cv.CV_64F, 0, 1, ksize=5)
# abs_sobely = np.abs(sobely)
# edgy = np.sqrt((sobelx**2 + sobely**2))
# # print(f"sqrt sqrds: {np.sqrt((sobelx**2 + sobely**2))!r}")

#fullsize IMAGE pass/draw
sobelx = cv.Sobel(img,cv.CV_64F, 1, 0, ksize=5)
abs_sobelx = np.abs(sobelx)
sobely = cv.Sobel(img,cv.CV_64F, 0, 1, ksize=5)
abs_sobely = np.abs(sobely)
edgy = np.sqrt((sobelx**2 + sobely**2))

plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Original, scaled.'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(edgy,cmap = 'gray')
plt.title('Combined Sobels'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(abs_sobelx,cmap = 'gray')
plt.title('Sobel X-axis'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(abs_sobely,cmap = 'gray')
plt.title('Sobel Y-axis'), plt.xticks([]), plt.yticks([])
plt.show()

(least_e, dirs) = least_edge(edgy)

print(f"leastE:{least_e!r}// dirs:{dirs!r}")

# Save the gradient magnitude image
# cv.imwrite("gradient_magnitudes.png", np.uint8(edgy))

# plt.subplot(2,2,1),plt.imshow(resized,cmap = 'gray')
# plt.title('Original, scaled.'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(edgy,cmap = 'gray')
# plt.title('Combined Sobels'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(abs_sobelx,cmap = 'gray')
# plt.title('Sobel X-axis'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(abs_sobely,cmap = 'gray')
# plt.title('Sobel Y-axis'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv.imshow("display image", resized)


