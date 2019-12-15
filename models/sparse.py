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
# from sklearn import preprocessing
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class MovingAverage():
#     def __init__(self, mu):
#         self.mu = mu
        
#     def __call__(self,x, last_average):
#         new_average = self.mu*x + (1-self.mu)*last_average
#         return new_average

class RoutingCoefficients(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self,routings, num_prime_filters):
    '''
    Calculates Routing Coefficients for the Latent Caps.
    1. Reshape routings from Nx1152x10 to Nx6x6x32x10
    2. Max over dim 3
    3. sum over dim 2 
    4. sum over dim 1
    return
    '''
    dim = int(math.sqrt(int(routings.shape[1]/num_prime_filters)))
    routings = routings.view([-1,dim,dim,num_prime_filters,routings.shape[2]])
    self.capsule_routing = routings.max(dim=3)[0]
    self.capsule_routing = self.capsule_routing.sum(dim=2)
    self.capsule_routing = self.capsule_routing.sum(dim=1)
    return self.capsule_routing

class WinningFrequency(nn.Module):
  '''
  Finds winning frequency of each latent capsule within each batch.
  1. Find number of times each capsule got rank 0 in a batch.
  '''
  def __init__(self):
    super().__init__()
    self.global_win_counts = 0

  def get_win_counts(self):
    return self.global_win_counts
  
  def reset_win_counts(self):
    self.global_win_counts = 0
  
  def forward(self, ranks):
    # ranks: [N,num_caps]

    #transpose, [num_caps,N]
    ranks = ranks.t()

    #count num. zeros row wise
    win_counts = (ranks==0).sum(dim=1)

    self.global_win_counts += win_counts
    
    #win freq = (num. wins)/ batch_size
    win_freq = win_counts.float()/ranks.shape[1]



    return win_freq

class MovingAverage(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self,shadow,variable, decay=0.99):
    if shadow.sum() == 0:
      return variable
    else:
      return decay * shadow + (1-decay) * variable

def boosting_op(ema, boosting_weights, num_caps=10, tgt_min_freq=4e-2, tgt_max_freq=1e-1, boost_factor=1e-1):
  def boost(c, weights, factor):
    weights[c] = weights[c] + factor
    return weights
  def deboost(c, weights, factor):
    weights[c] = weights[c] - factor
    weights[c] = torch.max(weights[c], torch.ones_like(weights[c]))
    return weights


  for ix,e in enumerate(ema):
    if e < tgt_min_freq:
      boosting_weights = boost(ix,boosting_weights,boost_factor)
    
    if e > tgt_max_freq:
      boosting_weights = deboost(ix,boosting_weights,boost_factor)
  
  return boosting_weights


class WeightedRouting(nn.Module):
  def __init__(self,num_caps=10,num_prime_filters=32,ema_decay=0.99, steepness_factor=12, clip_threshold=0.01):
    super().__init__()
    self.num_caps = num_caps
    self.num_prime_filters = num_prime_filters
    
    self.ema_decay = ema_decay
    self.gamma = steepness_factor
    self.clip = clip_threshold
    # self.ema = MovingAverage(ema_decay)

    self.freq_ema = torch.zeros(self.num_caps,dtype=torch.float,requires_grad=False).to(device)
    self.boosting_weights = torch.ones(self.num_caps, dtype=torch.float,requires_grad=False).to(device)

    self.rc = RoutingCoefficients()
    self.winning_freq = WinningFrequency()
    self.moving_avg = MovingAverage()

  def forward(self, routings):
    capsule_routings = self.rc(routings, self.num_prime_filters)

    #Boosting
    capsule_routings *= self.boosting_weights

    #Get ranks
    _, order = torch.topk(capsule_routings, self.num_caps)
    _, ranks = torch.topk(-order, self.num_caps)

    #Winning Frequency
    self.freq = self.winning_freq(ranks)
    # self.freq_ema = self.moving_avg(self.freq_ema, freq, self.ema_decay)
    self.freq_ema.data = self.moving_avg(self.freq_ema, self.freq, self.ema_decay)

    #Normalise
    norm_ranks = ranks.float() / (self.num_caps-1)
    routing_mask = torch.exp(-self.gamma * norm_ranks)

    routing_mask = routing_mask - (routing_mask * (routing_mask<self.clip))

    return routing_mask, ranks


class SparseMask(nn.Module):
  
  def __init__(self, caps_dim=16, num_caps=10, num_primary_filters=32, mode = 'norm'):
    super().__init__()
    
    self.dim = caps_dim
    self.num_caps = num_caps
    self.num_primary_filters = num_primary_filters
    self.mode = mode

    if mode.lower() == 'weighted-routing':
      self.masking = WeightedRouting( num_caps=self.num_caps, num_prime_filters = self.num_primary_filters)
     
    elif mode.lower() == 'routing':
      self.masking = RoutingCoefficients()
    elif mode.lower() == 'norm':
      pass
    else:
      raise NotImplementedError('Only norm, routing, weighted-routing supported.')

  def forward(self, routings,class_caps):
    if self.mode.lower() == 'norm':
      logits = torch.norm(class_caps,dim=-1)
      active_capsule = torch.argmax(logits,dim=-1)
      capsule_mask = F.one_hot(active_capsule, num_classes=self.num_caps)
      return capsule_mask
    elif self.mode.lower() == 'routing':
      routing_coeff = self.masking(routings, num_prime_filters=self.num_primary_filters)
      active_capsule = torch.argmax(logits,dim=-1)
      capsule_mask = F.one_hot(active_capsule, num_classes=self.num_caps)
      return capsule_mask
    elif self.mode.lower() =='weighted-routing':
      capsule_mask,_ = self.masking(routings)
      return capsule_mask
    else:
      raise NotImplementedError('Only norm, routing, weighted-routing supported.')




