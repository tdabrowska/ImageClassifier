# importing all packages

import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from torch import optim
from PIL import Image

#import functions
from user_input_arg import user_input_arg

def load_data():
    ''' Loads data from the users folder, returns dictionary with dataloaders (trainloader,
    validloader, testloader) and train_image_dataset
    '''    
    
    in_arg = user_input_arg()
#     print(in_arg)
    
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_dir_dict = {
        "train_dir": train_dir,
        "valid_dir": valid_dir,
        "test_dir": test_dir
    }
    
    return data_dir_dict


def preprocess_images():
    
    data_dir_dict = load_data()
    train_dir = data_dir_dict["train_dir"]
    valid_dir = data_dir_dict["valid_dir"]
    test_dir = data_dir_dict["test_dir"]
    
    train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    #Loading the datasets with ImageFolder
    train_image_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_image_dataset = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_image_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_image_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=64)
    
    dataloaders_dict = {
        "trainloader": trainloader,
        "validloader": validloader,
        "testloader": testloader,
        "train_image_dataset": train_image_dataset
    }
    
    return dataloaders_dict
    
    