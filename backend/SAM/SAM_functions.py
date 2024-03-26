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
    # Load the original image in RGB
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Prepare an RGBA image for the output: initialize alpha to 0 (fully transparent)
    rgba_image = np.zeros((*original_image.shape, 4), dtype=np.uint8)
    rgba_image[:, :, :3] = original_image  # Copy RGB values
    rgba_image[:, :, 3] = 0  # Set alpha to 0

    for mask_path in mask_paths:
        # Load the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Update the alpha channel of rgba_image based on the mask
        rgba_image[:, :, 3] = np.where(mask == 255, 255, rgba_image[:, :, 3])

    # Find the bounding box around the non-transparent areas
    coords = np.column_stack(np.where(rgba_image[:, :, 3] > 0))
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Crop the image to the bounding box
    cropped_rgba_image = rgba_image[y0:y1, x0:x1]

    # Save the resulting image with transparent background where masks are not applied
    result_path = os.path.join(output_path, 'segmented_and_cropped.png')  # Use PNG to keep transparency
    cv2.imwrite(result_path, cv2.cvtColor(cropped_rgba_image, cv2.COLOR_RGBA2BGRA))
    print(f"Cropped image with masks applied and transparent background saved to {result_path}")


        
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
