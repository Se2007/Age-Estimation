import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim

import numpy as np
import wandb

import dataset
import model
import utils 
import os

key_file = './wandb-key.txt'

if os.path.exists(key_file):
    with open(key_file) as f:
        key = f.readline().strip()
    wandb.login(key=key)
else:
    print("Key file does not exist. Please create the key file with your wandb API key.")

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load(model, loss, optimizer, device='cpu', reset = False, load_path = None):
    model = model
    loss_fn = loss
    optimizer = optimizer

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
            loss_fn.load_state_dict(sate['loss_fun'])
            optimizer.load_state_dict(sate['optimizer'])
            optimizer_to(optimizer, device)
    return model, loss_fn, optimizer
   


def save(save_path, model, optimizer, loss_fn):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_fun' : loss_fn.state_dict()
    }

    torch.save(state, save_path)

def plot(train_hist, valid_hist, label):
    print(f'\nTrained {len(train_hist)} epochs')

    plt.plot(range(len(train_hist)), train_hist, 'k-', label="Train")
    plt.plot(range(len(valid_hist)), valid_hist, 'y-', label="Validation")

    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.legend()
    plt.show()

info = {'num_epoch' :10,
        'lr' : 0.001,
        'weight_decay' : 0.001,
        'device' : 'cuda',
        'reset': True,
        'name_load' : 'model_loss5.148',
        'model_load_path' : './model/MI/',
        'model_save_path' : './model/MI/'
        }

seed = 3
wandb_enable = False


if wandb_enable:
    wandb_arg_name = input('Please input the WandB argument (run) name:')
    wandb.init(
        project='Age-estimation',
        name=wandb_arg_name,
        config={
            'lr': info['lr'],
            'weight_decay': info['weight_decay'],
            'num_epoch': info['num_epoch']
        }
    )



if __name__ == '__main__':

    loss_train_hist = []
    loss_valid_hist = []

    metric_train_hist = []
    metric_valid_hist = []

    load_path = info['model_load_path'] + info['name_load'] + ".pth"

    set_seed(seed)
    # model = model.Resnet(1, reset=info['reset'])
    # model = model.MultiInputModel(1)#  model.InputModel(1)

    layers = [4, 4, 8, 2]  # num of layers in the four blocks
    embed_dims = [192, 384, 384, 384]
    num_heads = [6, 12, 12, 12]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False] # do downsampling after first block
    outlook_attention = [True, False, False, False ]
    model = model.MIVolo(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 )

    loss_fn = nn.L1Loss()
    optimizer =  optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=3e-05) # optim.AdamW(model.parameters(), lr=info['lr'], weight_decay=info['weight_decay'])   


    model, loss_fn, optimizer = load(model, loss_fn, optimizer, device=info['device'], reset = info['reset'], load_path = load_path)

    set_seed(seed)
    train_loader = dataset.IMDB(mode='train')(batch_size=32)
    valid_loader = dataset.IMDB(mode='valid')(batch_size=84)

    # train_loader = dataset.UTKFace(mode='train')(batch_size=32)
    # valid_loader = dataset.UTKFace(mode='valid')(batch_size=124)
    # train_loader = dataset.CACD(train=True)(batch_size=124)
    # valid_loader = dataset.CACD(train=False)(batch_size=264)
    

    

    epochs = info['num_epoch']
    set_seed(seed)

    for epoch in range(1, epochs+1):

        # torch.backends.cudnn.benchmark = False
        _, loss_train, metric_train = utils.train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=epoch, device='cuda')
        loss_valid, metric_valid = utils.evaluate(model, valid_loader, loss_fn, device='cuda')

        
        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)

        metric_train_hist.append(metric_train)
        metric_valid_hist.append(metric_valid)


        print(f'Train      - Loss:{loss_train}  Metric:{metric_train}')
        print(f'Validation - Loss:{loss_valid}  Metric:{metric_valid}')
        print()

        if wandb_enable:
            wandb.log({"metric_train": metric_train, "loss_train": loss_train,
                        "metric_valid": metric_valid, "loss_valid": loss_valid})


    save_path = info['model_save_path'] + 'MImodel_loss' +f'{loss_train:.4}'+ ".pth"
    save(save_path, model, optimizer, loss_fn)

    plot(metric_train_hist, metric_valid_hist, "Metric")
    plot(loss_train_hist, loss_valid_hist, 'Loss')
    