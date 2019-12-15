import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import tqdm_notebook
import pandas as pd
import math
from collections import defaultdict

from models.layers import *
from models.sparse import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReconNet(nn.Module):
  def __init__(self, dim=16, n_lcaps=10):
    super().__init__()
    self.fc1 = nn.Linear(dim * n_lcaps, 512)
    self.fc2 = nn.Linear(512,1024)
    self.fc3 = nn.Linear(1024,1600)
    self.dim = dim
    self.n_lcaps = n_lcaps
    self.reset_params()

  def reset_params(self):
    nn.init.normal_(self.fc1.weight, 0, 0.1)
    nn.init.normal_(self.fc2.weight, 0, 0.1)
    nn.init.normal_(self.fc3.weight, 0, 0.1)

    nn.init.constant_(self.fc1.bias,0.1)
    nn.init.constant_(self.fc2.bias,0.1)
    nn.init.constant_(self.fc3.bias,0.1)

  def forward(self, x, target):
    # mask = Variable(torch.zeros((x.size()[0],self.n_lcaps)),requires_grad=False).to(device)
    # # mask = mask.float()
    # # import pdb; pdb.set_trace()
    # mask.scatter_(1,target.view(-1,1).long(),1.)
    # print(mask.shape)
    # mask = F.one_hot(target.long(),num_classes=self.n_lcaps)
    self.mask = target.unsqueeze(2)
    self.mask = self.mask.repeat(1,1,self.dim)

    x = x*self.mask
    x = x.view(-1,self.dim * self.n_lcaps)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x


class CapsNet(nn.Module):
  def __init__(self, n_lc=10):
    super().__init__()
    self.conv1 = nn.Conv2d(1,256,kernel_size=9,stride=1)
    self.primarycaps = PrimaryCaps(256,32,8,9,2) #686 output
    self.n_primary_caps = 32*6*6
    self.digitCaps = DenseCaps_v2(nc=n_lc,num_routes=self.n_primary_caps)

    self.decoder = ReconNet()
    self.caps_score = CapsToScalars()
  def forward(self,x, target):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.primarycaps(x)

    class_caps = self.digitCaps(x)

    probs = self.caps_score(class_caps)
    
    pred = torch.argmax(probs,dim=-1)

    if target is None:
      recon = self.decoder(class_caps, pred)
    else:
      recon = self.decoder(class_caps, target)

    # Matching return statement to DeepCaps to make things a bit easier...!
    return class_caps, None, recon, pred


class SparseCaps(nn.Module):
  def __init__(self, n_lc=10,caps_dim=16,sparse_mode='norm'):
    super().__init__()

    self.sparse_mode = sparse_mode
    self.winners = defaultdict(list)

    self.conv1 = nn.Conv2d(1,256,kernel_size=9,stride=1)
    self.primarycaps = PrimaryCaps(256,32,8,9,2) #686 output
    self.n_primary_caps = 32*12*12 # 32*6*6 for imSize 28, 32x12x12 for imsize 40
    self.digitCaps = DenseCaps_v2(nc=n_lc,num_routes=self.n_primary_caps, out_dim=caps_dim)

    self.sparse_mask = SparseMask(caps_dim=caps_dim, num_caps=n_lc, num_primary_filters=32, mode=self.sparse_mode)

    self.decoder = ReconNet(n_lcaps=n_lc)
    self.caps_score = CapsToScalars()


  def reset_winners(self):
    self.winners = defaultdict(list)

  def forward(self,x, target):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.primarycaps(x)

    class_caps, routings = self.digitCaps(x)

    probs = self.caps_score(class_caps)
    
    pred = torch.argmax(probs,dim=-1)

    mask = self.sparse_mask(routings, class_caps) # Nxnum_caps

    tgts = target.argmax(1)

    #Get Max Activation Capsule for each digit.
    #Used for logging purposes.
    # for tgt in tgts.unique():
    #   tgt_mask = mask[tgts==tgt]
    #   winner = tgt_mask.argmax(1)
    #   self.winners[tgt.cpu().detach().numpy()[0]] += list(winner.cpu().detach().numpy())
    recon = self.decoder(class_caps, mask)


    # Matching return statement to DeepCaps to make things a bit easier...!
    return class_caps, None, recon, pred

