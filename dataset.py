import time
from typing import Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numbers
import cv2
from tqdm import tqdm
import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import pandas as pd
from utils import AverageMeter


class MeanStd(object):
    def __init__(self):

        self.transform = T.Compose([
                    CustomPad(),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                ])

        self.df = pd.read_csv('./IMDB-clean dataset/12.csv')

        dataset = IMDBDataset(csv_file=self.df, transform=self.transform)

        self.data_loader = DataLoader(dataset, batch_size=712, shuffle=True)
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        face_mean = AverageMeter()
        face_std = AverageMeter()

        body_mean = AverageMeter()
        body_std = AverageMeter()
        
        with tqdm(self.data_loader, unit='cal') as tepoch:
            for face, body ,_,_,_,_ in tepoch:
                face = face.cuda()
                body = body.cuda()
                face_mean.update(face.mean(dim=(0,2,3)))
                face_std.update(face.std(dim=(0,2,3)))

                body_mean.update(body.mean(dim=(0,2,3)))
                body_std.update(body.std(dim=(0,2,3)))
        
        print(f'Face -> mean : {face_mean.avg} -- std : {face_std.avg}')
        print(f'Body -> mean : {body_mean.avg} -- std : {body_std.avg}')

class CustomPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        return F.pad(img, self.get_padding(img), fill=self.fill, padding_mode=self.padding_mode)

    def get_padding(self, image):
        w, h = image.size
        max_wh = max(w, h)
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = int(h_padding + 0.5) if h_padding % 1 != 0 else int(h_padding)
        t_pad = int(v_padding + 0.5) if v_padding % 1 != 0 else int(v_padding)
        r_pad = int(h_padding - 0.5) if h_padding % 1 != 0 else int(h_padding)
        b_pad = int(v_padding - 0.5) if v_padding % 1 != 0 else int(v_padding)

        return (l_pad, t_pad, r_pad, b_pad)

class AddGaussianNoise(object):
    def __init__(self, mean=0, stddev=.25):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, img):
        noisy_image = torch.clamp(
            img + torch.randn_like(img) * self.stddev + self.mean, 0, 255
        )#.byte()
        return noisy_image


'''  First version of IMDBDataset
class IMDBDataset(object):
    def __init__(self, csv_file, face_transform=None, body_transform=None) -> None:
        
        self.csv_file = csv_file
        self.face_transform = face_transform
        self.body_transform = body_transform
        self.gender_dict = {'M': 0, 'F': 1}
        
    def __len__(self):
      return len(self.csv_file)

    def __getitem__(self, idx):
        sample = self.csv_file.iloc[idx, :]

        face_point = tuple(int(i) for i in (sample.x_min, sample.y_min, sample.x_max, sample.y_max))
        body_point = (sample.body_x_min, sample.body_y_min, sample.body_x_max, sample.body_y_max)

        filename = sample.filename
        
        img = Image.open('./IMDB-clean dataset/imdb/'+filename).convert("RGB")

        face = self.face_transform(img.crop((face_point[0], face_point[1], face_point[2], face_point[3])))

        draw = ImageDraw.Draw(img)


        draw.rectangle([(face_point[0], face_point[1]), (face_point[2], face_point[3])], fill =(0, 0, 0, 128))
        body = self.body_transform(img.crop((body_point[0], body_point[1], body_point[2], body_point[3])))
        

        age = torch.tensor([sample.age], dtype=torch.float32)
        # gender = torch.tensor(self.gender_dict[sample.gender], dtype=torch.int32)

        return face, body, age#, gender, face_point, body_point'''
    
class IMDBDataset(Dataset):
    def __init__(self, root, csv_file, face_transform=None, body_transform=None):
        self.root_dir = root
        self.face_transform = face_transform
        self.body_transform = body_transform
        self.data = csv_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]

        img_name = sample.filename
        
        face = Image.open(self.root_dir+'/'+img_name+'/face.jpg')
        body = Image.open(self.root_dir+'/'+img_name+'/body.jpg')

        age = torch.tensor([sample.age], dtype=torch.float32)
        gender = torch.tensor([sample.gender], dtype=torch.int32)

        face = self.face_transform(face)
        body = self.body_transform(body)

        return face, body, age#, gender


class CACDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        file_path_list = glob.glob(root+'/*.jpg')
        self.data = []

        for f in file_path_list:
            age = torch.tensor(int(f.split('\\')[-1].split('_')[0]))
            
            self.data.append([f, age])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        path, age = self.data[index]

        if self.transform is not None:
            img = self.transform(Image.open(path))

        elif self.transform is None:
            img =  Image.open(path)

        # img = torch.from_numpy(cv2.imread(path)).permute(2,0,1).float()
#        print(img.shape)         img
        return img, age


class UTKDataset(Dataset):

    def __init__(self, root, csv_file, transform=None):
        self.root_dir = root
        self.csv_file = csv_file
        self.transform = transform
        self.data = csv_file
    #   self.data = pd.read_csv(csv_file)
        self.gender_dict = {'Male': 0, 'Female': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]

        img_name = sample.image_name
    #   img = Image.open(os.path.join(self.root_dir, img_name))
        face = Image.open(self.root_dir+'/'+img_name+'/face.jpg')
        body = Image.open(self.root_dir+'/'+img_name+'/body.jpg')

        age = torch.tensor([sample.age], dtype=torch.float32)
        gender = torch.tensor(self.gender_dict[sample.gender], dtype=torch.int32)
        ethnicity = sample.ethnicity

        face = self.transform(face)
        body = self.transform(body)
        


        return face, body, age

'''
class UTKDatasetNew(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        self.root = root
        file_path_list = os.listdir(root)
        self.data = []

        for f in file_path_list:
            age = torch.tensor(int(f.split('_')[0]))
            try:
                gen = torch.tensor([float(f.split('_')[1])])
            except:
                gen = torch.tensor([float(0)])
            
            self.data.append([f, age, gen])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        path, age, gen = self.data[index]

        if self.transform is not None:
            img = self.transform(Image.open(self.root+'/'+path+'/body.jpg'))

        elif self.transform is None:
            img = Image.open(self.root+'/'+path+'/face.jpg') 

        # img = torch.from_numpy(cv2.imread(path)).permute(2,0,1).float()
#        print(img.shape)         img
        return img, age, gen
'''


'''
class UTKDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        file_path_list = glob.glob(root+'/*.jpg')
        self.data = []

        for f in file_path_list:
            age = torch.tensor(int(f.split('\\')[-1].split('_')[0]))
            gen = torch.tensor([float(f.split('\\')[-1].split('_')[1])])
            
            self.data.append([f, age, gen])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        path, age, gen = self.data[index]

        if self.transform is not None:
            img = self.transform(Image.open(path))

        elif self.transform is None:
            img = Image.open(path) 

        # img = torch.from_numpy(cv2.imread(path)).permute(2,0,1).float()
#        print(img.shape)         img
        return img, age, gen
 '''   

class CACD(object):
    def __init__(self, train=True, mini=False) :
        self.mini = mini  
        
        RGB_MEAN = [ 0.485, 0.456, 0.406 ]
        RGB_STD = [ 0.229, 0.224, 0.225 ]

        self.transform = T.Compose([
                T.Resize((128,128)), 
                T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=15),
                T.RandomPerspective(distortion_scale=0.2, p=0.7),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean = RGB_MEAN, std = RGB_STD),
                # AddGaussianNoise(),
            ])

        self.path_dataset = "./CACD_croped"

        self.train = train
        

    def __call__(self, batch_size) :
        dataset = CACDataset(root=self.path_dataset, transform=self.transform)
        valid_dataset,train_dataset = random_split(dataset,(3000, len(dataset)-3000))

        if self.mini == False :
            if self.train :
                data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            else :
                data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        elif self.mini == True:
            dataset,_ = random_split(train_dataset,(1000, len(train_dataset)-1000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader


class IMDB(object):
    def __init__(self, mode, mini=False, gaussian_noise=False) :
        assert mode in ['train', 'valid', 'test'], 'mode should be train, test or valid'
        self.mini = mini  
        
        face_MEAN = [ 0.4260, 0.3192, 0.2682 ]
        face_STD = [ 0.3303, 0.2628, 0.2376 ]

        body_MEAN = [ 0.2061, 0.1810, 0.1685 ]
        body_STD = [ 0.2843, 0.2589, 0.2490 ]

        transform = [
                    # CustomPad(),
                    # T.Resize((224, 224)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.ToTensor(),
                ]
        
        face_transform = transform + [T.Normalize(mean = face_MEAN, std = face_STD)]
        body_transform = transform + [T.Normalize(mean = body_MEAN, std = body_STD)]
        if gaussian_noise:
            face_transform.append(AddGaussianNoise())
            body_transform.append(AddGaussianNoise())


        self.face_transform = T.Compose(face_transform)
        self.body_transform = T.Compose(body_transform)
        
        self.root = './IMDB-clean dataset/imdb_part'

        df = pd.read_csv(self.root + '/IMDB_dataset.csv')

        df, _= train_test_split(df, test_size=0.5, random_state=42)

        df_train, temp = train_test_split(df, test_size=0.3, random_state=42)# stratify=df.age,
        df_test, df_valid = train_test_split(temp, test_size=0.5, random_state=42)#  stratify=temp.age,

        if mode == 'train' :
            self.csv_file = df_train
        elif mode == 'valid':
            self.csv_file = df_valid
        elif mode == 'test' :
            self.csv_file = df_test

        

    def __call__(self, batch_size) :
        dataset = IMDBDataset(root=self.root, csv_file=self.csv_file, face_transform=self.face_transform, body_transform=self.body_transform)

        if self.mini == False :
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

        elif self.mini == True:
            dataset,_ = random_split(dataset,(1000, len(dataset)-1000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)##, num_workers=10

        return data_loader

class UTKFace(object):
    def __init__(self, mode, mini=False, gaussian_noise=False) :
        assert mode in ['train', 'valid', 'test'], 'mode should be train, test or valid'
        self.mini = mini  
        
        RGB_MEAN = [ 0.485, 0.456, 0.406 ]
        RGB_STD = [ 0.229, 0.224, 0.225 ]
        
        if not gaussian_noise:
            self.transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.ToTensor(),
                    T.Normalize(mean = RGB_MEAN, std = RGB_STD),
                ])
        else:
            self.transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.ToTensor(),
                    T.Normalize(mean = RGB_MEAN, std = RGB_STD),
                    AddGaussianNoise(),
                ])
            

        self.path_dataset = './UTKFace_inthewild/crop'
        df = pd.read_csv('./UTKFace_inthewild/utkface_dataset.csv')

        df_train, temp = train_test_split(df, test_size=0.3, stratify=df.age, random_state=42)
        df_test, df_valid = train_test_split(temp, test_size=0.5, stratify=temp.age, random_state=42)

        if mode == 'train' :
            self.csv_file = df_train
        elif mode == 'valid':
            self.csv_file = df_valid
        elif mode == 'test' :
            self.csv_file = df_test

        

    def __call__(self, batch_size) :
        dataset = UTKDataset(root=self.path_dataset, csv_file=self.csv_file, transform=self.transform)

        if self.mini == False :
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

        elif self.mini == True:
            dataset,_ = random_split(dataset,(1000, len(dataset)-1000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader


if __name__=='__main__':
    # dataset = UTKFace('train')
    # dataloader = dataset(110)
    # print(next(iter(dataloader))[0].shape)


    # df = pd.read_csv('./IMDB-clean dataset/12.csv')
    # d = IMDBDataset(df, T.Compose([CustomPad(),T.Resize((224, 224)),T.ToTensor(),]),  T.Compose([CustomPad(),T.Resize((224, 224)),T.ToTensor(),])).__getitem__(89)
    # d[0].show()
    # d[1].show()
    # print(d[1:])


    dataset = IMDB('train')
    dataloader = dataset(11)
    print(len(next(iter(dataloader))))

    # MeanStd()()



