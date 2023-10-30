import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from PIL import Image
import cv2

import dataset
import model

def load(model,  device='cpu', reset = False, load_path = None):
    model = model

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
            
    return model

def accuracy(model, data_loader):
    acc_list = []
    for img, age,_ in data_loader:
        age_p = model(img)
        _,perd = age_p.max(1)
        correct = perd.eq(age).sum().item() / img.shape[0]
        acc_list.append(correct)
    
    return sum(acc_list) / len(acc_list)

from torchvision import transforms as T

RGB_MEAN = [ 0.485, 0.456, 0.406 ]
RGB_STD = [ 0.229, 0.224, 0.225 ]

transform = T.Compose([
                T.Resize((128,128)), 
                T.ToTensor(),
                T.Normalize(mean = RGB_MEAN, std = RGB_STD),
            ])



if __name__=='__main__':
    mae = nn.L1Loss()
    data_loader = dataset.CACD(train=False)(batch_size=284)#dataset.UTKFace(train=False)(batch_size=164)


    load_path = './model/' + 'new_dataset_model_loss0.07474' + ".pth"

    model = model.Resnet(120, reset=False)
    model = load(model, device='cpu', load_path = load_path)
    model.eval()
    # model = torch.load()

    imr = cv2.imread('./14_Aaron_Johnson_0001.jpg')
    # im = cv2.cvtColor(imr, cv2.COLOR_RGB2BGR)
    im = Image.fromarray(cv2.cvtColor(imr, cv2.COLOR_BGR2RGB))
    # print(transform(torch.FloatTensor(im).permute(-1, 1, 0)).unsqueeze(0).shape)
    # print(model(transform(torch.FloatTensor(im).permute(-1, 1, 0)).unsqueeze(0)).max(1))

    # im = transform(Image.open('./14_Aaron_Johnson_0001.jpg'))
    im = transform(im)
    print(model(im.unsqueeze(0)).max(1))

    img, age = next(iter(data_loader))
    # print(age[0].shape, img[0].shape)
    print(img.shape)

    age_p = model(img)
    _,perd = age_p.max(1)
    print(f'MAE : {mae(perd.float(), age.float())}')
    correct = perd.eq(age).sum().item()
    print(correct)
    # print(f'Accuracyy : {accuracy(model, data_loader)}')

    print(f'Target : {age[10]}  --  Periction : {age_p.max(1)[1][10]}')#age_p.argmax(dim=1).item()}')
    plt.imshow(img[10].permute(1,-1,0))
    # plt.imshow(im)#permute(1,-1,0)
    plt.show()