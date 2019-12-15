import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
# from keras.utils import to_categorical
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import tqdm_notebook
import pandas as pd
from sklearn import preprocessing
import math

from tensorboardX import SummaryWriter

writer = SummaryWriter()


def write_image_summary()