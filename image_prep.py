from PIL import Image, ImageTk
from Xlib import display

image = Image.open("ascii/enceladusstripes_cassini_forBG.jpg")
scale_factor = float(
    (display.Display().screen().width_in_pixels / image.size[0]))

def scale_me(image, scale_factor):
  width, height = image.size
  new_width = int(width * scale_factor)
  new_image = image.resize((new_width, int(height * scale_factor)))
  new_image.show()
  return new_image

scale_me(image, scale_factor)
