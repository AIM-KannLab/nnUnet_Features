import onnxruntime as rt
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="1"

providers = ['AzureExecutionProvider', 'CPUExecutionProvider']

onnx_model = rt.InferenceSession("/media/sdb/onnx/nnTransferModel.onnx", providers=providers)
path_images = "/media/sdb/preproc-images/nnunet/imagesTs"

path_images_list = []
for item in os.listdir(path_images):
  path_images_list.append(os.path.join(path_images, item))

num_images = len(path_images_list)

path_images_list = sorted(path_images_list)


feature_list = []
for i in range(0, num_images, 2):

  # Create batch
  batch_paths = path_images_list[i:i+2]
  
  # Preprocess batch
  input_data = np.empty((2, 1, 96, 192, 192))
  for j, path in enumerate(batch_paths):
    img = nib.load(path)
    volume = img.get_fdata()
    volume = (volume - volume.mean()) / volume.std()
    resampled = ndimage.zoom(volume, (96/volume.shape[0], 192/volume.shape[1], 192/volume.shape[2])) 
    input_data[j] = resampled[np.newaxis, :]

    # Pass batch through model
  input_data = input_data.astype(np.float32)
  onnx_outputs = onnx_model.run(["output"], {"input": input_data})[0]
  
  # Save features
  for j, path in enumerate(batch_paths):
    feature = onnx_outputs[j] 
    feature_list.append(feature)

df_features = pd.DataFrame(feature_list)
df_features.to_csv('features_nnunet.csv', index=True)
