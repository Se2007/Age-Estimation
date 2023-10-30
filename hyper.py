from torch import optim
from model import MultiInputModel, InputModel, MIVolo
import dataset
from utils import train_one_epoch
from prettytable import PrettyTable
from torch import nn
from colorama import Fore, Style, init

device = 'cuda'

train_loader  = dataset.IMDB(mode='train', mini=True)(batch_size=32)

num_epochs = 5

 
learning_rates = [1e-6, 3e-5, 1.5e-5, 3e-4, 3e-3, 1e-3,  3e-2]
weight_decays = [1e-2, 3e-2, 1e-3, 1e-4, 1e-5, 5e-5, 1e-6]



loss_list = []

best_lr = None
best_wd = None
best_loss = float('inf')  
min_num = float('inf')
second_min = float('inf')

table = PrettyTable()
table.field_names = ["LR \ WD"] + [f"WD {i}" for i in weight_decays]


for lr in learning_rates:
    for wd in weight_decays:
    
        print(f'\nLR={lr}, WD={wd}')

        loss_fn = nn.L1Loss()
        model = MIVolo().to(device)#  InputModel(1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


        for epoch in range(1, num_epochs+1):
            model, loss, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device=device)

        loss_list.append(float(f'{loss:.4f}'))

   

sorted_list = sorted(loss_list)
first_min = sorted_list[0]
second_min = sorted_list[1]

first_min_idx = loss_list.index(first_min)
second_min_idx = loss_list.index(second_min)

loss_list[first_min_idx] = f"{Fore.GREEN}{first_min}{Fore.WHITE}"
loss_list[second_min_idx] = f"{Fore.YELLOW}{second_min}{Fore.WHITE}"
loss_list = list(map(str, loss_list))



o = 0

for i in learning_rates:
    row = [f"LR {i}"]

    losses = loss_list[o:len(weight_decays)+o]
    o += len(weight_decays)

    row += losses
    table.add_row(row)


print(table)