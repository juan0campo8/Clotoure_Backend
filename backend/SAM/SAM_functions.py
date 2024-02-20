import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import json
import os
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
        masks = mask_generator.generate(image)

        print('writing to file')



        for x in range(len(masks)):
            filename = path + r'\mask'+ str(x) + '.jpg'
            
            data = im.fromarray(masks[x]['segmentation'])
            data.save(filename)
            #with open(filename, "w") as outfile: 
            #    json.dump(masks[x], outfile)
        os.remove(ifile)
    except Exception as e:
        os.remove(ifile)
        print(str(e))


