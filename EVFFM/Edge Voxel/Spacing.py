import os
import SimpleITK as sitk

def resave_mask(image_path, mask_path, output_mask_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Ensure that the mask has the same direction, spacing, and origin as the image
    mask.SetDirection(image.GetDirection())
    mask.SetSpacing(image.GetSpacing())
    mask.SetOrigin(image.GetOrigin())

    # Save the updated mask
    sitk.WriteImage(mask, output_mask_path)

# Get input paths from the user
image_dir = input("Enter the path to the 'images' directory: ")
num_masks = int(input("Enter the number of 'masks' directories: "))
masks_dirs = [input(f"Enter path for mask directory {i + 1}: ") for i in range(num_masks)]

# Process each specified mask directory
for mask_dir in masks_dirs:
    output_mask_dir = os.path.join(os.path.dirname(mask_dir), os.path.basename(mask_dir) + "_spacing")

    # Create the output directory if it doesn't exist
    os.makedirs(output_mask_dir, exist_ok=True)

    # Iterate through each file in the image directory
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        output_mask_path = os.path.join(output_mask_dir, filename)

        # Check if the corresponding mask file exists before processing
        if os.path.exists(mask_path):
            resave_mask(image_path, mask_path, output_mask_path)
