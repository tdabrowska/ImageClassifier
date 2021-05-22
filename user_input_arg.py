# PROGRAMMER: Teresa Dabrowska
# DATE CREATED: 01.05.2021                                  
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --data_dir with default value 'flowers'
#     2. CNN Model Architecture as --arch with default value 'densenet'
#     3. Directory to save checkpoints as --save_dir with default value 'checkpoint.pth'
#     4. Json file with images category names as --category_names with default value 'cat_to_name.json'
#     5. Device to choose GPU/CPU as --user_device with default value 'gpu'
#     6. Hyperpatameters: Learning Rate as --learning_rate with default value '0.0025'
#     7. Hyperpatameters: Hidden Units as --hidden_units with default value '256'
#     8. Hyperpatameters: Epochs as --epochs with default value '3'
#     9. Image directory as --img_dir with default value 'flowers/test/16/image_06657.jpg'
#     10. Top K classes to display as --top_k with default value '3'

# import modules
import argparse

def user_input_arg():
    # Creating Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Creating command line arguments using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type = str, default = 'flowers',
                        help = 'path to the folder of your images')
    parser.add_argument('--arch', type = str, default = 'densenet',
                        help = 'the CNN model architecture to use: vgg, densenet or alexnet')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth',
                        help = 'path to file to save checkpoints')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'name of the json file with images names')
    parser.add_argument('--user_device', type = str, default = 'GPU',
                        help = 'the device to use: GPU, CPU')
    parser.add_argument('--learning_rate', type = float, default = 0.0025,
                        help = 'model hyperparameter: learning rate')
    parser.add_argument('--hidden_units', type = int, default = 256,
                        help = 'model hyperparameter: numer of hidden units')
    parser.add_argument('--epochs', type = int, default = 3,
                        help = 'model hyperparameter: numer of epochs')
    parser.add_argument('--img_dir', type = str, default = 'flowers/test/16/image_06657.jpg',
                        help = 'path to the image to predict')
    parser.add_argument('--top_k', type = int, default = 3,
                        help = 'top K must likely classes to display')
    
    return parser.parse_args()