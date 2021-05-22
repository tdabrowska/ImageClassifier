import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from torch import optim
from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    with Image.open(image) as pil_image:
#         print(pil_image.size)
        width, height = pil_image.size
        #resizing the image to 256 px shortest side, keeping the aspect ratio
        if width < height:
            pil_image.thumbnail((256, (height/ (width/256))))
        else:
            pil_image.thumbnail((width/(height/256), 256))
        
#         print(pil_image.size)
        
        #center crop 224x224 px
        width, height = pil_image.size
        left = int((width-224) / 2)
        upper = int((height-224) / 2)
        right = int(left + 224)
        lower = (upper + 224)
        box = (left, upper, right, lower)
#         print(box)
        pil_image = pil_image.crop(box)
#         print(pil_image.size)
        pil_image.save('new_image.jpg')
        
    
    np_image = np.array(pil_image) / 255
    
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np_image = np.transpose(np_image, (2,0,1))

    
    torch_image = torch.from_numpy(np_image)
    
    #return np_image
    return torch_image.float()