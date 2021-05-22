# PROGRAMMER: Teresa Dabrowska
# DATE CREATED: 01.05.2021                                  
# REVISED DATE: 
# PURPOSE:  The program predicts image category name along with the probability of that category. The user passes
#           in a single image and the program returns the image category name with top K class probability. 
#           The trained model is loaded from the checkpoint. The user can set the image directory, the number  
#           of top must likely classes to dipslay, the device (CPU/GPU) and the file with category names.
# 
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --img_dir <path to the image to predict> --top_k <top K must likely classes to display> 
#                       --category_names <file that contains images names> --user_device <device to use GPU/CPU>
#   Example call:
#    python train.py --img_dir --top_k 3 --category_names cat_to_name.json --user_device gpu


# import all packages
import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from torch import optim
from PIL import Image
import numpy as np
import json

#import functions
from user_input_arg import user_input_arg
from load_data import *
from process_image import process_image
from model import build_model

in_arg = user_input_arg()

image_path = in_arg.img_dir
user_topk = in_arg.top_k
cat_dir = in_arg.category_names
model_dir = in_arg.save_dir

model_dict = build_model()
model = model_dict["model"]

with open(cat_dir, 'r') as f:
        cat_to_name = json.load(f)

def predict(image_path, model, topk=user_topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
        
    #use GPU if it is a user choice and it is available, if not use CPU        
    if in_arg.user_device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # process image
    image = process_image(image_path)
    image.unsqueeze_(0)
    
    # load model from checkpoint file
    state_dict = torch.load(model_dir)
    model.load_state_dict(state_dict)
    
    model.eval()

# Calculate the class probabilities (softmax) for img
    class_to_idx = torch.load('class_to_idx.pth')
    

    with torch.no_grad():
        output = torch.exp(model(image.to(device)))
        top_p, top_class = output.topk(topk, dim=1)
        top_p = top_p.tolist()[0]
        top_class = top_class.tolist()[0]
        
        idx_to_classes = {value:key for key, value in class_to_idx.items()}
        top_classes = []
        for idx in top_class:
            top_classes.append(idx_to_classes[idx])
    
    return top_p, top_classes


top_p, top_classes = predict(image_path, model)

top_classes_names = []
for idx in top_classes:
    top_classes_names.append(cat_to_name[idx])

    
print('Prediction result:')
for i in range(0, user_topk):
    print(f'{i+1} top klass: {top_classes_names[i]} with probability: {top_p[i]:.4f}')