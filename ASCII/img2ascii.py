import os

from PIL import Image


def scale_image(image, new_width=100):
    (original_width, original_height) = image.size
    aspect_ratio = original_height/float(original_width)
    new_height = int(aspect_ratio * new_width)
    new_image = image.resize((new_width, new_height))
    return new_image

def grayscale_image(image):
    return image.convert('L')

def map_pixels_to_ascii(image, range_width=25):
    ascii_str = ''
    ascii_chars = '@%#*+=-:. '
    for pixel_value in image.getdata():
        ascii_str += ascii_chars[pixel_value//range_width]
    return ascii_str

def convert_image_to_ascii(image_path, output_path=None):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(e)
        return
    image = scale_image(image)
    image = grayscale_image(image)

    ascii_str = map_pixels_to_ascii(image)
    ascii_str_len = len(ascii_str)
    ascii_img=""
    for i in range(0, ascii_str_len, image.width):
        ascii_img += ascii_str[i:i+image.width] + "\n"

if output_path is None:
  output_path = os.path.join(os.getcwd(), 'ascii_image.txt')

# Save the ASCII art to a text file
with open(output_path, 'w') as f:
  f.write(ascii_img)
exit()
# Replace 'image_path' with the path of the image you want to convert
convert_image_to_ascii('ascii/enceladusstripes_cassini_forBG.jpg')
