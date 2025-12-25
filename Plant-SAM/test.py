import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry_baseline
from segment_anything import sam_model_registry, SamPredictor
import warnings
warnings.filterwarnings('ignore')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 182/255, 193/255, 0.7])  

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=4))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'.jpg',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.jpg',bbox_inches='tight',pad_inches=-0.1)
    plt.close()



if __name__ == "__main__":
  
    sam_checkpoint = "./plant_sam.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    input_path = "./Input"
    file_list = os.listdir(input_path)


    for file in file_list:
        if file.endswith('.jpg'):
            image_name = os.path.splitext(file)[0]
            image_path = os.path.join(input_path, file)
            print("image:   ",file)

            ssq_token_only = False
            # input_point = np.array([[495,518],[217,140]])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_box = np.array([[66.00,274.00,236.00,384.00]])  ## 框输入
            input_point, input_label = None, None
            predictor.set_image(image)

            batch_box = False if input_box is None else len(input_box)>1 
            result_path = './Output/'
            os.makedirs(result_path, exist_ok=True)

            if not batch_box: 
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box = input_box,
                    multimask_output=False,
                    hq_token_only=ssq_token_only , 
                )

                show_res(masks,scores,input_point, input_label, input_box, result_path + image_name, image)

            else:
                masks, scores, logits = predictor.predict_torch(
                    point_coords=input_point,
                    point_labels=input_label,
                    boxes=transformed_box,
                    multimask_output=False,
                    hq_token_only=ssq_token_only ,
                )
                masks = masks.squeeze(1).cpu().numpy()
                scores = scores.squeeze(1).cpu().numpy()
                input_box = input_box.cpu().numpy()
                show_res_multi(masks, scores, input_point, input_label, input_box, result_path + image_name, image)


        

    