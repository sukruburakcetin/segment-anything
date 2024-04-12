# Let's correct the function to set the outside of the circle to be transparent
import cv2
import numpy as np


def remove_black_make_transparent(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Get image dimensions
    height, width, channels = img.shape

    # Assuming the circle is at the center and the image is square, calculate the radius
    radius = min(height, width) // 2

    # Create a mask with the same dimensions as the image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw a filled circle in the mask at the center with the calculated radius
    cv2.circle(mask, (width // 2, height // 2), radius, (255), thickness=-1)

    # Create an all white image
    white_img = 255 * np.ones_like(img)

    # Copy only the masked area (the circle) from the original image
    circular_img = np.where(mask[..., None] == 255, img, white_img)

    # Set the alpha channel to the mask (0 for transparent, 255 for opaque)
    circular_img = cv2.cvtColor(circular_img, cv2.COLOR_BGR2BGRA)
    circular_img[:, :, 3] = mask

    # Save the image with the transparency
    transparent_circular_img_path = 'transparent_circular_image.png'
    cv2.imwrite(transparent_circular_img_path, circular_img)

    return transparent_circular_img_path


img_path = "WE0NXXN1_equirectangular90_fisheye.png"
# Apply the function to make the background transparent
transparent_circle_image_path = remove_black_make_transparent(img_path)
