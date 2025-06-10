import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage

# Define input folders for original images and ROI masks
img_folder = './images'
roi_folder = './masks'

# Get all .nii.gz files in the respective folders
img_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.nii.gz')]
roi_files = [os.path.join(roi_folder, f) for f in os.listdir(roi_folder) if f.endswith('.nii.gz')]

# Prompt user to input dilation range, e.g., "1-5"
expand_range_input = input("Enter the dilation range (e.g., '1-5'): ")
start_range, end_range = map(int, expand_range_input.split('-'))

# Prompt user to input dilation step size
expand_step = int(input("Enter the dilation step size (e.g., '1'): "))

# Check for valid range
if start_range >= end_range or start_range < 0:
    raise ValueError("Invalid range. The start must be less than the end and non-negative.")

# Loop through all image-mask pairs
for i in range(len(img_files)):
    # Read original image and mask
    img_path = img_files[i]
    roi_path = roi_files[i]
    
    image = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(roi_path)
    mask_array = sitk.GetArrayFromImage(mask)

    # Apply dilation with different iterations
    for iteration in range(start_range, end_range + 1, expand_step):
        # Perform binary dilation
        dilated_mask_array = ndimage.binary_dilation(mask_array, iterations=iteration).astype(mask_array.dtype)

        # Compute the border (dilated - original)
        border_mask_array = dilated_mask_array - mask_array

        # Define output directories for this dilation iteration
        output_img_dir = os.path.join('./expanded_images', f'iter_{iteration}')
        output_mask_dir = os.path.join('./expanded_masks', f'iter_{iteration}')

        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        # Save original image
        output_img_path = os.path.join(output_img_dir, os.path.basename(img_path))
        sitk.WriteImage(image, output_img_path)

        # Save dilated edge mask
        output_mask_path = os.path.join(output_mask_dir, os.path.basename(roi_path))
        sitk.WriteImage(sitk.GetImageFromArray(border_mask_array), output_mask_path)
