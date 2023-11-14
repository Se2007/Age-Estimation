from typing import Any
import cv2
import glob
import os
import torch
import time
import numbers
import pandas as pd
import numpy as np
from face_detection import RetinaFace
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms as T



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




class CropImg(object):
    def __init__(self, root) :
        self.detector = RetinaFace()

        file_path_list = glob.glob(root+'/*.jpg')
        self.data = []

        self.transform_with_padding = T.Compose([
                                            CustomPad(),
                                            T.Resize((224, 224)),
                                            # T.ToTensor(),
                                                    
                                            ])

        for f in file_path_list:
            name = f.split('\\')[-1]
            
            self.data.append([f, name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        path, name = self.data[index]

        img = cv2.imread(path)  #BGR
        # img = Image.open(path)

        return img, name   

    def pad(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        img = self.transform_with_padding(pil_image)
        img = np.array(img)
        # cv2.imshow("img2_padded", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   
        
    def __call__(self, root_save):
        root = root_save


        for i in range(2197, self.__len__()):
            print(i)
            
            img, name = self.__getitem__(i)
            name = name.split('.')[0]

            print(img.shape)
            

            if img is not None:
                faces = self.detector(img)
                box, landmarks, score = faces[0]
                x, y, w, h = map(int, (box[0], box[1], box[2], box[3]))
                print(x,y,w,h,'  ',score)

                    
                    
                cut = img[y:h, x:w]
                # print(cut.shape)
                if 0 not in cut.shape:
                    cut = self.pad(cut)
                    # cv2.imshow('sepehr', cut)

                    os.chdir(root)

                    os.mkdir(name)
                    os.chdir(name)
                
                    cv2.imwrite('face.jpg', cut)
                            
                    body = cv2.rectangle(img, (x, y), (w, h), (0, 0, 0), -1)
                    body = self.pad(body)
                    cv2.imwrite('body.jpg', body)


                    os.chdir(len(root.split('/')) * '../')

                    
class CSV(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        # Initialize lists to store data
        self.image_names = []
        self.ages = []
        self.ethnicities = []
        self.genders = []

        
        
    def __call__(self, csv_filename) :
        
        for filename in os.listdir(self.dataset_dir):
            try:
                parts = filename.split('_')
                # Format: [age]_[gender]_[ethnicity]_[other_info].jpg

                if len(parts) < 4:
                    print(filename)
                    continue
                # print(parts, filename)
                age = int(parts[0])
                gender = 'Male' if int(parts[1]) == 0 else 'Female'
                ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(parts[2])]
                
                if age > 80:
                    continue

                self.image_names.append(filename)
                self.ages.append(age)
                self.ethnicities.append(ethnicity)
                self.genders.append(gender)
            except:
                continue

        # Create a DataFrame from the lists
        data = {
            'image_name': self.image_names,
            'age': self.ages,
            'ethnicity': self.ethnicities,
            'gender': self.genders
        }
        df = pd.DataFrame(data)

        # Save DataFrame to CSV
        csv_filename = csv_filename
        df.to_csv(csv_filename+'/utkface_dataset.csv', index=False)

        print(f"CSV file '{csv_filename}' created successfully.")

       


if __name__ == '__main__':

    ### for crop dataset
    '''
    crop = CropImg('./UTKFace_inthewild/part3')
    print(crop.__len__())
    crop('./UTKFace_inthewild/part3_crop')
    '''

    ### make csv file from dataset

    csv = CSV('./UTKFace_inthewild/crop')
    csv('./UTKFace_inthewild')







