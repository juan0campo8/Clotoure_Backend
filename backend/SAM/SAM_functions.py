import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import json
import os
import datetime as time
from PIL import Image as im
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def print_func():
    return "success"
    

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


def generate_image(ifile, fn):
    try:    
        name = fn.rsplit('.', 1)[0]
        cwd = os.getcwd() 
        print(cwd)
        path = os.path.join(cwd, name)
        print(path)
        os.mkdir(path)
        print(ifile)
        image = cv2.imread(ifile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print('reading image')
        sam_checkpoint = os.getcwd() + r'\SAM\sam_vit_h_4b8939.pth'
        model_type = "vit_h"

        device = "cpu"
        # set to "cpu"
        print('loading model')
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        print('setting up mask generation')
        mask_generator = SamAutomaticMaskGenerator(sam)

        print('generating masks')
        begin = time.datetime.now()
        print(begin)
        masks = mask_generator.generate(image)

        print('writing to file')
        end = time.datetime.now()
        print(end)
        print(end - begin)

        # updated code (Adding segmentation to the original image, allowing user to select which masks)
        mask_files = []
        for x, mask in enumerate(masks):
            filename = os.path.join(path, f'mask{x}.jpg')
            mask_files.append(filename)
            data = im.fromarray(mask['segmentation'])
            data.save(filename)
            
        # List masks for the user
        print("Available masks:")
        for idx, mask_file in enumerate(mask_files):
            print(f'{idx}: {mask_file.split(os.sep)[-1]}')
        
        # Prompt user for selection
        user_input = input("Enter the numbers of the masks to apply, separated by commas (e.g., 0,2,3): ")
        selected_indices = [int(idx) for idx in user_input.split(',')]
        selected_mask_paths = [mask_files[idx] for idx in selected_indices]

        # Apply selected masks
        segment_image_with_selected_masks(ifile, selected_mask_paths, path)

    except Exception as e:
        os.remove(ifile)
        print(str(e))
def segment_image_with_selected_masks(original_image_path, mask_paths, output_path):
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create an accumulator mask to hold the combined mask of the selected indices
    accumulator_mask = np.zeros(original_image.shape[:2], dtype=np.float32)
    
    for mask_path in mask_paths:
        # Read the mask and convert it to a floating point type
        mask = cv2.imread(mask_path, 0).astype(np.float32) / 255.0

        # Apply Gaussian blur to the mask to smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Accumulate the mask
        accumulator_mask = np.maximum(accumulator_mask, mask)

    # Threshold the accumulated mask to create a binary mask
    _, binary_mask = cv2.threshold(accumulator_mask, 0.5, 1.0, cv2.THRESH_BINARY)
    
    # Convert the binary mask back to boolean
    binary_mask = binary_mask.astype(bool)

    segmented_image = np.zeros_like(original_image)
    for i in range(3):  # For each color channel
        segmented_image[:, :, i][binary_mask] = original_image[:, :, i][binary_mask]
    
    # Save or Display the Result
    result_path = os.path.join(output_path, 'segmented_image.jpg')
    cv2.imwrite(result_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved to {result_path}")


if __name__ == "__main__":
    generate_image('path/to/your/image.jpg', 'filename.jpg')
