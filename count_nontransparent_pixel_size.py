from PIL import Image
import numpy as np

# Load the image
img_path = 'transparent_circular_image.png'
image = Image.open(img_path)

# Convert the image to RGBA if it is not already in that format
image = image.convert("RGBA")

# Extract the alpha channel as a numpy array
alpha_channel = np.array(image)[:, :, 3]

# Count the non-transparent pixels (where alpha channel is not 0)
non_transparent_pixel_count = np.count_nonzero(alpha_channel != 0)

print("size", non_transparent_pixel_count)
