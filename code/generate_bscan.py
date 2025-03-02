import numpy as np
import cv2
import torch
from PIL import Image
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from net import *

path_ascan  = "pathtoascan"
mean_a_scan = np.load(path_ascan, allow_pickle=True)
mean_a_scan = np.log(mean_a_scan + 1e-6)  # Apply log transform for better range
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Normalize to [-1,1] and convert to tensor
mean_a_scan = cv2.resize(mean_a_scan, (1024, 1024), interpolation=cv2.INTER_LINEAR)
mean_a_scan = (mean_a_scan - mean_a_scan.min()) / (mean_a_scan.max() - mean_a_scan.min())*2 - 1
mean_a_scan = torch.from_numpy(mean_a_scan).float().unsqueeze(0)
model = torch.load("pathtomodel", weights_only=False, map_location=torch.device("cpu"))
model.eval()
with torch.no_grad():
    start = time.time()
    generated_b_scan =model(mean_a_scan.unsqueeze(0)).squeeze(0).squeeze(0).cpu()

# Convert tensors to NumPy for visualization
print('Image Generation Time: ' + str(time.time() - start))
generated_b_scan_np = generated_b_scan.numpy()
image = Image.fromarray(generated_b_scan_np)
image.save(path_ascan[:-4] + ".tiff")
