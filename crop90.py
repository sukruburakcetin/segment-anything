# Load the new image
from PIL import Image

img_path_new = 'WE0NXXN1_equirectangular.jpg'
img_new = Image.open(img_path_new)

# Calculate the new dimensions for the new image
width_new, height_new = img_new.size
new_height_new = height_new // 2

# Crop the image (left, upper, right, lower) for the new image
cropped_img_new = img_new.crop((0, 0, width_new, new_height_new))

# Save the cropped image with high quality
cropped_img_path_new_high_quality = 'cropped_image_upper_half_new_high_quality.jpg'
cropped_img_new.save(cropped_img_path_new_high_quality, quality=100)