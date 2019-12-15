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
import datetime
from collections import OrderedDict


from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from models.networks import DeepSparseCaps
from models.layers import one_hot
from models.sparse import boosting_op

now = datetime.datetime.now()
writer = SummaryWriter(f'logs/{now.date()}/{now.hour}_{now.minute}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mse_loss = nn.MSELoss(reduction='none')

def margin_loss(x, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
  v_c = torch.norm(x, dim=2, keepdim=True)
  tmp1 = F.relu(m_plus - v_c).view(x.shape[0], -1) ** 2
  tmp2 = F.relu(v_c - m_minus).view(x.shape[0], -1) ** 2
  loss_ = labels*tmp1 + lamda*(1-labels)*tmp2
  loss_ = loss_.sum(dim=1)
  return loss_
    
def reconst_loss(recnstrcted, data):
  loss = mse_loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
  return loss.sum(dim=1)
    
def loss(x, recnstrcted, data, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
  # print(type(recnstrcted))
  loss_ = reconst_loss(recnstrcted, data)#+ margin_loss(x, labels, lamda, m_plus, m_minus)
  return loss_.mean()

def accuracy(indices, labels):
  correct = 0.0
  for i in range(indices.shape[0]):
      if float(indices[i]) == labels[i]:
          correct += 1
  return correct


def write_summaries(loss,inputs,recon,model, ep):
    ep += 1
    # Write Loss
    writer.add_scalar('loss/recon',loss,ep)

    # Write Sample images
    idxs = np.random.permutation(len(inputs))[:5]
    recon = recon.squeeze().reshape(-1,40,40)
    recon = recon.unsqueeze(1)
    to_write = torch.cat((inputs[idxs],recon[idxs]))
    grid = make_grid(to_write,nrow=5, range=(0,1))
    writer.add_image('reconstruction', grid, ep)

    # Write Winning Frequencies
    wins = model.sparse_mask.masking.winning_freq.global_win_counts
    for ix,win in enumerate(wins):
        writer.add_scalar(f'wins/capsule_{ix}', win, ep)
    
    # Write Weights of DenseCaps Layer
    writer.add_histogram('deepcaps/weights',model.digitCaps.W)

    # Write Routing Coefficients for the Latent Capsules
    coeffs = model.sparse_mask.masking.rc.capsule_routing.t()
    for ix,coeff in enumerate(coeffs):
      writer.add_histogram(f'CapsRoutings/capsule_{ix}',coeff,ep)
    
    # winners = model.winners
    # import pdb; pdb.set_trace()
    # winners = OrderedDict(sorted(winners.items()))
    # for digit in winners:
    #   writer.add_histogram(f'MaxContribution/digit_{digit}',winners[digit],ep)


def write_sp_summaries(model, step):
    ema = model.sparse_mask.masking.freq_ema
    boost_wts = model.sparse_mask.masking.boosting_weights

    for ix, m in enumerate(ema):
        writer.add_scalar(f'ema/capsule_{ix}', m, step)
    
    for ix, wt in enumerate(boost_wts):
        writer.add_scalar(f'boost_wt/capsule_{ix}', wt, step)

    for ix,freq in enumerate(model.sparse_mask.masking.freq):
      writer.add_histogram(f'win_freq/capsule_{ix}',freq,step)
    # print(model.sparse_mask.masking.freq.sum(dim=-1))

def train(train_loader, model, num_epochs, lr=0.001, batch_size=64, lamda=0.5, m_plus=0.9,  m_minus=0.1, boost_every=-1):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    lambda1 = lambda epoch: 0.5**(epoch // 10)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    step = 0
    for epoch in tqdm(range(num_epochs)):
      model.sparse_mask.masking.winning_freq.reset_win_counts()
      model.reset_winners()
      loss_track=0.
      for batch_idx, (data, label_) in enumerate(train_loader):
        step += 1
        data = data.float().to(device)
        labels = one_hot(label_.to(device))
        optimizer.zero_grad()
        outputs, masked, recnstrcted, indices = model(data, labels)
        
        loss_val = loss(outputs, recnstrcted, data, labels, lamda, m_plus, m_minus)
        loss_track += loss_val.item()
        
        if boost_every > 0 and batch_idx%boost_every == 0:
          # model.sparse_mask.boosting_weights.data =  boosting_op(model.sparse_mask.freq_ema,model.sparse_mask.boosting_weights)
          # model.sparse_mask.boosting_weights.data =  boosting_op_og(model.sparse_mask.freq_ema,model.sparse_mask.boosting_weights)
          model.sparse_mask.masking.boosting_weights = boosting_op(model.sparse_mask.masking.freq_ema,model.sparse_mask.masking.boosting_weights,16)
          write_sp_summaries(model, step)
        
        loss_val.backward()
        optimizer.step()
      write_summaries(loss=loss_track,inputs=data,recon=recnstrcted, model=model, ep=epoch)
      torch.save(model,f'./saved/sparse_caps_40_EP_{epoch}_{now.date()}_{now.hour}_{now.minute}.pth')
      print('\n')
      print(f'EP {epoch}, Loss: {loss_track}')
      # print(model.sparse_mask.masking.freq_ema.data)
      print('WIN COUNTS\n')
      print(model.sparse_mask.masking.winning_freq.get_win_counts())
      loss_track=0.
      # lr_scheduler.step()



def main():
    batch_size = 128
    num_epochs = 100
    lamda = 0.5
    m_plus = 0.9
    m_minus = 0.1


    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(8), transforms.RandomCrop(40),
                              transforms.ToTensor()
                          ])),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(8), transforms.RandomCrop(40),
                              transforms.ToTensor()
                          ])),
            batch_size=batch_size, shuffle=True)

    model = DeepSparseCaps(n_lc=16,sparse_mode='weighted-routing').to(device)

    train(train_loader,model,50,batch_size=128,boost_every=15)


if __name__ == '__main__':
    main()