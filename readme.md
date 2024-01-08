# Astronomy Picture of the Day (APOD) Desktop Background Setter

This Python program fetches the Astronomy Picture of the Day (APOD) from NASA's website and sets it as your desktop background.

## Dependencies

The program uses the following Python libraries:

- `ctypes`
- `os`
- `platform`
- `tkinter`
- `io`
- `requests`
- `bs4 (BeautifulSoup)`
- `PIL (Pillow)`
- `Xlib`

You can install these dependencies using pip:

```bash
pip install ctypes os platform tkinter io requests bs4 pillow python-xlib

## How It Works

1. The program first checks the operating system and gets the screen dimensions.
2. It then sends a GET request to the APOD website and parses the HTML response with BeautifulSoup.
3. It finds the image tag and gets the source URL.
4. It finds the first `<p>` tag after the image and gets its text content.
5. It requests the image and opens it with PIL.
6. If the image height exceeds the screen height, it adjusts the messagebox height.
7. It displays the image in a preview window.
8. It prompts the user to confirm before setting the image as the desktop background.
9. If the user confirms, it saves the image to a file and sets the image as the desktop background.

## Usage

To run the program, simply execute the Python script:

```bash
python apod.py
```

Enjoy your new APOD desktop background!
```
