        
# PROGRAMMER: Teresa Dabrowska
# DATE CREATED: 01.05.2021                                  
# REVISED DATE: 
# PURPOSE: Training the network on the user data set. The trained model will be save as a checkpoint. 
#          With this program we can choose 3 different pretrained CNN model architectures to check 
#          which provides the 'best' classification. The user can set the hyperparameters (learning rate, 
#          numer of hidden units, training epochs) and the device (CPU/GPU). The program is printing out
#          the training loss, validation loss and validation accuracy as the network trains.

# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --data_dir <directory with images> --arch <model>
#               --save_dir <directory to save checkpoints> --category_names <file that contains images names>
#               --user_device <device to use GPU/CPU> --learning_rate <hyperparameter value of learning rate>
#               --hidden_units <hyperparameter number of hidden_units> --epochs <hyperparameter number of epochs>
#   Example call:
#    python train.py --dir flowers/ --arch vgg --save_dir checkpoint.pth --category_names cat_to_name.json --user_device GPU --learning_rate 0.0025 --hidden_units 256 --epochs 3


# importing all packages
import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from torch import optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

#import functions
from user_input_arg import user_input_arg
from load_data import *
from model import build_model

in_arg = user_input_arg()

cat_dir = in_arg.category_names

model_dict = build_model()

model = model_dict["model"]
optimizer = model_dict["optimizer"]
criterion = model_dict["criterion"]

with open(cat_dir, 'r') as f:
    cat_to_name = json.load(f)
    
def train_model():
    
    #loading data
    data_dir_dict = load_data()
    
    #preprocessing images
    dataloaders_dict = preprocess_images()
    trainloader = dataloaders_dict["trainloader"]
    validloader = dataloaders_dict["validloader"]
    testloader = dataloaders_dict["testloader"]
    
    #use GPU if it is a user choice and it is available, if not use CPU
    if in_arg.user_device == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #moving model to the default device
    model.to(device);

    #seting parameters
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            #moving inputs, labels to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch: {epoch} | "
                        f"Train loss: {running_loss/print_every:.2f} | "
                        f"Validation loss: {valid_loss/len(validloader):.2f} | "
                        f"Validation accuracy: {accuracy/len(validloader):.2f}")
                    running_loss = 0
                    model.train()

    print('end')        

    # Mapping of classes to indices
    train_image_dataset = dataloaders_dict["train_image_dataset"]
    #print(train_image_dataset.class_to_idx)
    model.class_to_idx = train_image_dataset.class_to_idx

    # Saving the checkpoint 
    file_dir = in_arg.save_dir
    torch.save(model.state_dict(), file_dir)
    torch.save(model.class_to_idx, 'class_to_idx.pth')
    torch.save(optimizer.state_dict, 'checkpoint_optim.pth')

train_model()