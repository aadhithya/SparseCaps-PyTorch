import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import numpy as np

class AffNISTData(Dataset):
  def __init__(self,dataset_path,ix=0):
    super().__init__()
    
    self.xData = np.load(dataset_path+f'_images_{ix}.npy').astype(np.float)
    self.yData = np.load(dataset_path+f'_labels_{ix}.npy').astype(np.int)
    
    self.xData /= 255.0

  def __len__(self):
    return len(self.xData)
   
  def __getitem__(self, idx):
    return np.reshape(self.xData[idx],[1,40,40]),self.yData[idx]#to_categorical(self.yData[idx],num_classes=10)