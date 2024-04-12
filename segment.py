import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"


# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    # Further, you can get the device count and device name
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
image_path = 'transparent_circular_image.png'
img = cv2.imread(image_path)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)
# print(len(masks))
# print(masks[0].keys())


# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show()


# import supervision as sv
#
# mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
# detections = sv.Detections.from_sam(masks)
# annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
#
# sv.plot_images_grid(
# images=[image, annotated_image],
# grid_size=(1, 2),
# titles=['source image', 'segmented image']
# )



mask_predictor = SamPredictor(sam)
mask_predictor.set_image(image)
# Provide points as input prompt [X,Y]-coordinates
# input_point = np.array([[2040, 0]]) # for equirectangular
# input_point = np.array([[1020, 0]]) # for equirectangular 90
# input_point = np.array([[2048, 1024]]) # for fisheye
input_point = np.array([[1024, 515]]) # for fisheye
# input_point = np.array([[1040, 1023]]) # for fisheye cropped
# input_point = np.array([[2048, 2000]]) # for equirectangular bottom
input_label = np.array([1])


# Predict the segmentation mask at that point
masks, scores, logits = mask_predictor.predict(
point_coords=input_point,
point_labels=input_label,
multimask_output=False,
)

image_transparent_temp = Image.open(image_path)

# Convert the image to RGBA if it is not already in that format
image_transparent_temp = image_transparent_temp.convert("RGBA")

# Extract the alpha channel as a numpy array
alpha_channel = np.array(image_transparent_temp)[:, :, 3]

# Count the non-transparent pixels (where alpha channel is not 0)
non_transparent_pixel_count = np.count_nonzero(alpha_channel != 0)

# Find the index of the highest value
index_of_highest_value = np.argmax(scores)

print("Index of the highest value:", index_of_highest_value)

# Calculate the pixel count (area) of the mask
mask_area = np.count_nonzero(masks[index_of_highest_value])
print("mask_area: ", mask_area)

print("sky express ratio: ", mask_area / non_transparent_pixel_count)
percentage = (mask_area / non_transparent_pixel_count) * 100
print("sky express percentage(%): ", percentage)

# print("image_area: ", image.shape[0]*image.shape[1])
# print("sky express ratio: ", mask_area/(image.shape[0]*image.shape[1]))
# percentage = (mask_area / (image.shape[0]*image.shape[1])) * 100
# print("sky express percentage(%): ", percentage)
# print("masks shape", masks.shape)  # (number_of_masks) x H x W

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()


# def visualize_mask_overlay(image, mask, alpha=0.5):
#     """
#     Visualize the transparent overlay of the mask on a blank canvas.
#
#     Args:
#     - image: The original image.
#     - mask: The binary mask to be visualized.
#     - alpha: The transparency level of the mask overlay. Default is 0.5.
#     """
#     # Create a blank canvas
#     overlay = np.zeros_like(image)
#
#     # Set the overlay pixels to red where the mask is non-zero
#     overlay[mask == 1] = [255, 0, 0]  # Red color
#
#     # Apply transparency to the overlay
#     overlay = (alpha * overlay).astype(np.uint8)
#
#     # Display the overlay
#     plt.figure(figsize=(10, 10))
#     plt.imshow(overlay)
#     plt.title("Transparent Overlay of Mask")
#     plt.axis('off')
#     plt.show()
#
#
# # Visualize the transparent overlay of the mask
# visualize_mask_overlay(image, masks[index_of_highest_value])

# def remove_masked_area(image, mask):
#     """
#     Remove the areas highlighted by the mask from the original image.
#
#     Args:
#     - image: The original image.
#     - mask: The binary mask highlighting the areas to be removed.
#
#     Returns:
#     - The image with the masked areas removed.
#     """
#     # Copy the original image to avoid modifying it directly
#     result = np.copy(image)
#
#     # Set the pixels in the mask area to black (remove them)
#     result[mask == 1] = [0, 0, 0]  # Black color
#
#     return result
#
#
# # Remove the areas highlighted by the mask from the original image
# image_with_removed_mask = remove_masked_area(image, masks[index_of_highest_value])
#
# # Display the resulting image
# plt.figure(figsize=(10, 10))
# plt.imshow(image_with_removed_mask)
# plt.title("Image with Masked Areas Removed")
# plt.axis('off')
# plt.show()
#
# # Save the resulting image
# cv2.imwrite('image_with_removed_mask.jpg', cv2.cvtColor(image_with_removed_mask, cv2.COLOR_RGB2BGR))

