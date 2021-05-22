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
    
def build_model():
    ''' Builds a model from one of 3 pretrained torchvision models, chosen by the user,
        returns a dictionary with model, optimizer and criterion
    '''
    
    in_arg = user_input_arg()
    print(in_arg)
    cat_dir = in_arg.category_names
    
    with open(cat_dir, 'r') as f:
        cat_to_name = json.load(f)

    # Creating dictionary with 3 available models to choose
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet121(pretrained=True)
    alexnet = models.alexnet(pretrained=True)

    models_dict = {'vgg': vgg16, 'densenet': densenet, 'alexnet': alexnet}
    
    model_name = in_arg.arch
    model = models_dict[model_name]

    #Building the model
    for param in model.parameters():
        param.requires_grad = False
        
    #user hyperparameters
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    number_of_outputs = len(cat_to_name)
    
    #seting number of inputs according to the chosen model
    if model_name == 'vgg':
        number_of_inputs = 25088
    elif model_name == 'densenet':
        number_of_inputs = 1024
    else:
        number_of_inputs = 9216
    
    classifier = nn.Sequential(nn.Linear(number_of_inputs, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_units,number_of_outputs),
                            nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model_dict = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion
    }
    
    return model_dict