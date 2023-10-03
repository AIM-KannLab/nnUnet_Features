**Goal: Extract the feature of images from the nnUNet architecture**

# Operating System
getNNUnet has been tested on Linux (Ubuntu 22.04)! It should work out of the box!

# Hardware requirements
We support GPU (recommended) and CPU

# Installation instructions
We strongly recommend that you install nnU-Net in a virtual environment! Pip or anaconda (e.g. pip or conda install) are both recommened.

Use a recent version of Python! 3.9 or newer is guaranteed to work!

**nnU-Net v2 can coexist with nnU-Net v1! Both can be installed at the same time.**

To install nnU-Net, follow the instructions on the following website: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md.

Try to work with the environment that you ran nnUNet and install these extra libraries:

pip install nnunetv2 

pip install onnx 

pip install torchinfo 

pip install onnx2torch 

pip install onnxruntime 

The first step is saving the model to onnx model. This getNNUnet.py does this in this repo. You should change these parts in the code.

1) TestInputModelPath:
Which is the path to nnUNet trained model(usually saved in the nnUNet folder), something with a name like nnUNetTrainerV2__nnUNetPlansv2.1

2) TestOutputPath:
The path you want to save the onnx model

3) onnxPath = TestOutputPath+'ONNX_MODEL/nnunetModel.onnx' 

The next step is to modify nnUNet and customize nnUNet, which allows slicing the model, adding layers, freezing weights, overriding input shape, and exporting the modified model to a new ONNX file. This is done by modiNNUET.py. You should change these parts in the code:

1) Change GPU based on your needs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

2) modelPathIN:
You have this path from onnxPath in getNNUnet.py

3) outputFilenamesIN:
It is your desired path for saving the model as onnx file

4) Change this line based on the feature vector size you need.
   
   For example, it gives us one score:
   
addModelIN = torch.nn.Sequential(torch.nn.AdaptiveMaxPool3d((6, 6, 6)), torch.nn.Flatten(), torch.nn.Linear(69120,4096), torch.nn.ReLU(), torch.nn.Linear(4096, 512), torch.nn.ReLU(), torch.nn.Linear(512, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1), torch.nn.Sigmoid())

It gives us a vector with a size of 4096:

addModelIN = torch.nn.Sequential(torch.nn.AdaptiveMaxPool3d((6, 6, 6)), torch.nn.Flatten(), torch.nn.Linear(69120,4096), torch.nn.Sigmoid())

Then the last step is extracting features of the nnUNet segmentation model for the medical images. This is done by feature_extraction.py. In this step, the extracted features are normalized. You should change these parts in the code.

1) onnx_model 
Give the path to the modified nnUNet model, which is the same as outputFilenamesIN from the previous step.

2) Change the GPU based on your needs
os.environ["CUDA_VISIBLE_DEVICES"]="0"

3) Path_images
Give the path of the images you tried to segment, which are all in one folder and in NIFTI formats.

The features are saved in feature_list and then saved as a CSV file with the name features_nnunet.csv, which has shape (number of patients, feature vector size). The features are saved as the order of the images in path_image.

# Notes:
* Maybe it is needed to install extra libraries like pandas, scipy, nibabel.
* The code is written for batch size equals to 2 based on the high use of GPU, and it is better not to change it. 
* Here the extracted features are related to the latent feature after Encoder, which is layer 37.

# Limitations:

* Only work with 3D 
* We assumed that your model are trained using CUDA
