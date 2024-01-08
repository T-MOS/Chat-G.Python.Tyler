import ctypes
import os
import platform
import tkinter as tk
from io import BytesIO
from tkinter import messagebox, scrolledtext

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageTk
from Xlib import display

# os == 'Windows' --> Get screen dimensions for "..."
if platform.system() == 'Windows':
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

# os == 'Linux' --> Get screen dimensions for "..."
if platform.system() == 'Linux':
    screen = display.Display().screen()
    screen_width = screen.width_in_pixels
    screen_height = screen.height_in_pixels

# os == 'Darwin' ...Mac... --> make tk/grab screen dimensions for "..."
if platform.system() == 'Darwin':
  def get_screen_size():  
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height
  get_screen_size()  
  screen_width, screen_height = get_screen_size()
  print(screen_width, screen_height)
  
# URL of APOD website
url = 'https://apod.nasa.gov/apod/astropix.html'

# Send GET request to APOD website and parse HTML response with BeautifulSoup
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')

# Find image tag and get source URL
img_tag = soup.find('img')
img_url = 'https://apod.nasa.gov/apod/' + img_tag['src']

# Find the first <p> tag after the image and get its text content
description = img_tag.find_next('p').text.strip()

# Request the image and open it with PIL
image_response = requests.get(img_url)
image = Image.open(BytesIO(image_response.content))

# Adjust messagebox height if it exceeds screen height
if image.height > screen_height:
  max_description_height = screen_height - image.height - 100  # Adjust the padding as needed
  max_description_lines = max_description_height // 20  # Assuming each line is around 20 pixels
lines = description.splitlines()
concatenated_description = ''
current_line_length = 0
for line in lines:
  line = line.strip()  # Remove whitespace at the beginning and end of the line
  if line.startswith('Explanation:' + '\n'):
    concatenated_description += '\n' + line
    current_line_length = len(line)
  else:
    line = line.replace(
        '\n', ' ')  # Replace line breaks within the line with a space
    if current_line_length + len(line) > screen_width // 10:
      concatenated_description += '\n' + line
      current_line_length = len(line)
    else:
      concatenated_description += line
      current_line_length += len(line)

# Display the image in a preview window
root = tk.Tk()
root.title('APOD Image Preview')

# Set the width of the scrolledtext widget to screen width
scroll_text = scrolledtext.ScrolledText(root,
                                        width=screen_width // 10,
                                        height=10,
                                        wrap=tk.WORD)
scroll_text.pack(fill=tk.BOTH, expand=True)
scroll_text.insert(tk.END, concatenated_description)

# Create a label and display the image
image_label = tk.Label(root)
image_label.pack()
photo = ImageTk.PhotoImage(image)
image_label.configure(image=photo)

# Prompt user to confirm before setting the image as desktop background
messagebox.showinfo('Image Preview', f'Image URL: {img_url}')
response = messagebox.askquestion(
    'Set Desktop Background', 'Set this image as your desktop background?')

if response == 'yes':
  # Save image to a file
  image_path = os.path.join(os.getcwd(), 'apod_image.jpg')
  image.save(image_path)

  # Set the image as desktop background
  SPI_SETDESKWALLPAPER = 0x0014
  ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0,
                                             image_path, 3)

  messagebox.showinfo('Set Background Successful',
                      'Desktop background has been set.')
else:
  messagebox.showinfo('Set Background Declined',
                      'Desktop background has not been changed.')

root.mainloop()
