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
    
    return original_image_rgba

def generate_image(ifile, fn):
    try:    
        name = fn.rsplit('.', 1)[0]
        cwd = os.getcwd() 
        path = os.path.join(cwd, name)
        os.mkdir(path)
        image = cv2.imread(ifile)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_checkpoint = os.getcwd() + r'\SAM\sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        device = "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)

        masks = mask_generator.generate(image_rgb)  # Assume this returns a list of masks

        composite_images_paths = []
        for x, mask in enumerate(masks):
            mask_applied_image = apply_mask_to_image(image_rgb, mask['segmentation'])
            composite_image_path = os.path.join(path, f'composite{x}.png')
            im.fromarray(mask_applied_image).save(composite_image_path)
            composite_images_paths.append(composite_image_path)
            
        # List the composite images for the user to select
        print("Available composites:")
        for idx, composite_path in enumerate(composite_images_paths):
            print(f'{idx}: {composite_path.split(os.sep)[-1]}')

        user_input = input("Enter the numbers of the composites to keep, separated by commas (e.g., 0,2,3): ")
        selected_indices = [int(idx) for idx in user_input.split(',')]
        selected_composite_paths = [composite_images_paths[idx] for idx in selected_indices]

        print("Selected composites:")
        for path in selected_composite_paths:
            print(path)
            # You might want to move these selected composites to a different directory or process them further as needed

    except Exception as e:
        print(str(e))
        if os.path.exists(ifile):
            os.remove(ifile)


        
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
