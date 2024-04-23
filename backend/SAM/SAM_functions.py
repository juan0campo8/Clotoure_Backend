import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import json
import io
import os
import datetime as time
from PIL import Image as im
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from COS import COS_functions as cos


def print_func():
    return "success"

def bytes_file(file):
    f = io.BytesIO()
    f.write(file)
    f.seek(0)    

def to_bucket(rgb, mask, folder, typ):
    data = im.fromarray(apply_mask_to_image(rgb, mask)).convert("RGBA")
    byts = io.BytesIO()
            
    data.save(byts, format="PNG")
    byts.seek(0)
    cos.upload_to_folder(folder, f'{typ}_mask.png', 'clotoure', byts)

def apply_mask_to_image(original_image, mask):
    """
    Apply a mask to the original image, making the masked area visible and the rest transparent.
    """
    # Ensure the mask is boolean
    mask_bool = mask.astype(bool)
    
    # Convert original image to RGBA if not already
    if original_image.shape[2] == 3:  # RGB
        original_image_rgba = np.concatenate([original_image, np.full((*original_image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
    else:  # Already RGBA
        original_image_rgba = original_image
    
    # Apply mask
    original_image_rgba[~mask_bool] = (0, 0, 0, 0)  # Set unmasked area to transparent
    
    print(type(original_image_rgba))

    return original_image_rgba

def generate_image(folder, frnt, bck):
    try:
        '''    
        name = fn.rsplit('.', 1)[0]
        cwd = os.getcwd() 
        print(cwd)
        path = os.path.join(cwd, name)
        print(path)
        os.mkdir(path)
        print(ifile)
        '''
        print(type(frnt))

        front = cv2.imdecode(np.asarray(bytearray(frnt.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
        front_rgb = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)

        back = cv2.imdecode(np.asarray(bytearray(bck.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
        back_rgb = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print('reading image')
        sam_checkpoint = os.getcwd() + r'\SAM\sam_vit_b_01ec64.pth'
        model_type = "vit_b"

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
        print('beginning front')
        front_masks = mask_generator.generate(front)
        print('beginning back')
        back_masks = mask_generator.generate(back)

        print('writing to file')
        end = time.datetime.now()
        print(end)
        print(end - begin)
        
        # updated code (Adding segmentation to the original image, allowing user to select which masks)
        #mask_files = []
        score = 0
        final_front = None
        for x, mask in enumerate(front_masks):
            #filename = os.path.join(path, f'mask{x}.jpg')
            #mask_files.append(filename)
            if score == 0:
                score = mask['predicted_iou']
                final_front = mask['segmentation']
            if mask['predicted_iou'] > score:
                final_front = mask['segmentation']


        to_bucket(front_rgb, final_front, folder, "front")

        score = 0
        final_back = None

        for x, mask in enumerate(back_masks):
            #filename = os.path.join(path, f'mask{x}.jpg')
            #mask_files.append(filename)
            #mask_files.append(filename)
            if score == 0:
                score = mask['predicted_iou']
                final_back = mask['segmentation']
            if mask['predicted_iou'] > score:
                final_back = mask['segmentation']

        to_bucket(back_rgb, final_back, folder, "back")
            
        # List the composite images for the user to select
        '''print("Available Masks:")
        for idx, composite_path in enumerate(composite_images_paths):
            print(f'{idx}: {composite_path.split(os.sep)[-1]}')

        user_input = input("Enter the numbers of the Masks to keep, separated by commas (e.g., 0,2,3): ")
        selected_indices = [int(idx) for idx in user_input.split(',')]
        
        # Calculate paths for the selected composite images
        selected_composite_paths = [composite_images_paths[idx] for idx in selected_indices]

        # Combine selected composite images into one and save it
        combine_selected_composites(selected_composite_paths, path)  # 'path' is the directory where you've been saving everything
        '''
    except Exception as e:
        print(str(e))
        

def find_bounding_box(image):
    """
    Find the bounding box of non-transparent pixels in an image.

    Args:
    image (PIL.Image): An image.

    Returns:
    The bounding box as a tuple (left, upper, right, lower).
    """
    # Convert the image to a numpy array
    np_image = np.array(image)
    
    # Find all non-transparent pixels
    non_transparent_pixels = np.argwhere(np_image[:, :, 3] > 0)
    
    # Find the bounding box coordinates
    upper_left = non_transparent_pixels.min(axis=0)
    lower_right = non_transparent_pixels.max(axis=0) + 1  # +1 because slice indices are exclusive at the top
    
    # Return as (left, upper, right, lower)
    return (upper_left[1], upper_left[0], lower_right[1], lower_right[0])

def combine_selected_composites(selected_paths, output_path):
    """
    Combine selected composite images into one, crop to the mask edges, and save it.

    Args:
    selected_paths (list of str): Paths to the selected composite images.
    output_path (str): Path where the combined and cropped image should be saved.
    """
    base_image = None

    for path in selected_paths:
        composite_image = im.open(path).convert("RGBA")

        if base_image is None:
            base_image = composite_image
        else:
            base_image = im.alpha_composite(base_image, composite_image)
    
    base_image = im.open()

    # Find the bounding box of non-transparent pixels
    bbox = find_bounding_box(base_image)

    # Crop the image to the bounding box
    cropped_image = base_image.crop(bbox)

    # Save the combined and cropped image
    combined_image_path = os.path.join(output_path, 'combined_selected_composites_cropped.png')
    cropped_image.save(combined_image_path)
    print(f"Combined and cropped image saved to {combined_image_path}")

        
# The following function is tentative and has not been tested yet
# {TODO} Test the following function for implementation and expand it to include 
# point inputs
#'''
def generate_manual_mask(ifile, fn, input_points, input_labels, input_box):
    try:
        name = fn.rsplit('.', 1)[0]
        cwd = os.getcwd() 
        path = os.path.join(cwd, name)
        os.mkdir(path)
        print(ifile)
        image = cv2.imread(ifile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_checkpoint = os.getcwd() + r'\SAM\sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        print('loading models')
        #onnx_model_path = os.getcwd() + r"\SAM\sam_vit_h_onnx_model.onnx"
        #onnx_model = SamOnnxModel(sam, return_single_mask=True)
        #ort_session = onnxruntime.InferenceSession(onnx_model_path)
        print('starting predictor')

        sam.to(device='cpu')
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        #image_embedding = predictor.get_image_embedding().cpu().numpy()
        #image_embedding.shape
        print('checking for segmentatino')
        
        if(input_points == None):
            print('segmenting')
            input_box = np.array(input_box)
            mask, _, _ = predictor.predict(
                point_coords= None,
                point_labels= None,
                box=input_box[None, :],
                multimask_output= False)
            print('creating new image')
            filename = os.path.join(path, 'mask1.jpg')
            data = im.fromarray(mask[0])
            data.save(filename)
            print("mask created")
            #egment_image_with_selected_masks(ifile, filename, path)
        elif(input_box == None):
            mask, _, _ = predictor.predict(
                point_coords= input_points,
                point_labels= input_labels,
                box= None,
                multimask_output= False)

            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) 

            filename = os.path.join(path, 'mask1.jpg')
            data = im.fromarray(mask_image)
            data.save(filename)        
        else:
            print("failure")
        


    except Exception as e:
        os.remove(ifile)
        print(str(e))
#'''

if __name__ == "__main__":
    generate_image('path/to/your/image.jpg', 'filename.jpg')
