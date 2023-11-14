import pandas as pd
from dataset import CustomPad
from tqdm import tqdm
import torch
import cv2
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T



os.chdir('./IMDB-clean dataset')


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
        name = sample.filename.split('/')[1].split('.')[0]

        face_point = tuple(int(i) for i in (sample.x_min, sample.y_min, sample.x_max, sample.y_max))
        body_point = (sample.body_x_min, sample.body_y_min, sample.body_x_max, sample.body_y_max)

        filename = sample.filename
        
        img = Image.open('./imdb/'+filename).convert("RGB")

        face = self.face_transform(img.crop((face_point[0], face_point[1], face_point[2], face_point[3])))

        draw = ImageDraw.Draw(img)


        draw.rectangle([(face_point[0], face_point[1]), (face_point[2], face_point[3])], fill =(0, 0, 0, 128))
        body = self.body_transform(img.crop((body_point[0], body_point[1], body_point[2], body_point[3])))
        

        age = sample.age
        gender = self.gender_dict[sample.gender]

        return name, face, body, age, gender
    


def crop_body_face_csv(root_save):
    root = root_save

    name_list = [] 
    age_list = []
    gender_list = [] 

    csv_file = pd.read_csv('./part_of_IMDB(body+face).csv')
    face_MEAN = [ 0.4260, 0.3192, 0.2682 ]
    face_STD = [ 0.3303, 0.2628, 0.2376 ]

    body_MEAN = [ 0.2061, 0.1810, 0.1685 ]
    body_STD = [ 0.2843, 0.2589, 0.2490 ]

    transform = [
        CustomPad(),
        T.Resize((224, 224)),
       # T.ToTensor(),
        ]
        
    face_transform = T.Compose(transform) #+ [T.Normalize(mean = face_MEAN, std = face_STD)]
    body_transform = T.Compose(transform) #+ [T.Normalize(mean = body_MEAN, std = body_STD)]

    dataset = IMDBDataset(csv_file, face_transform,  body_transform)


    for i in tqdm(range(dataset.__len__()), desc="Saving images", unit='cal'):

        name, face, body, age, gender= dataset.__getitem__(i)

        os.chdir(root)

        os.mkdir(name)
        os.chdir(name)
            
        face.save('face.jpg')
        body.save('body.jpg')

        os.chdir(2 * '../')
        # print(age)
        # print(gender)

        name_list.append(name)
        age_list.append(age)
        gender_list.append(gender)


    # Create a DataFrame from the lists
    data = {
        'filename': name_list,
        'age': age_list,
        'gender': gender_list
        }
    df = pd.DataFrame(data)

    df.to_csv(root+'/IMDB_dataset.csv', index=False)

    print(f"CSV file created successfully.")



        



'''
step one

list_name = ['train', 'test', 'valid']

for n in list_name:
    df = pd.read_csv(f'imdb_{n}_new.csv')


    l = []


    for i in  range(len(df)):#
        d = df.filename[i]
        if int(d.split('/')[0]) <= 39 :
            l.append(d)
        else:
            df.drop(i, inplace=True)
    print(len(l))
    df.to_csv(f'{n}.csv', index=False)
'''

'''
##step two

df1 = pd.read_csv('valid1.csv')
df2 = pd.read_csv('test1.csv')
df3 = pd.read_csv('train1.csv')

data_frames = [df1, df2, df3]

combined_df = pd.concat(data_frames, ignore_index=True)

combined_df.to_csv('12.csv', index=False)#'''





'''### filename,age,gender,x_min,y_min,x_max,y_max,head_roll,head_yaw,head_pitch

l = ['./imdb/38/nm2679438_rm3695806976_1989-8-15_2009.jpg', 20, 'M', 31, 39, 61, 77, 7.786003215506801, 1.7970911090239363, 2.2217050506208564]
i = ['./imdb/13/nm0000313_rm1851819008_1949-12-4_2010.jpg',61,'M',139,71,184,137,-0.9748775391392523,-10.860459236548833,26.80608852701976]


img = cv2.imread(i[0])

reg = cv2.rectangle(img, (i[3], i[4]), (i[5], i[6]), (0, 0, 0), 2)

cv2.imshow('fr2ame', reg)

# cv2.imshow('frame', img)

cv2.waitKey()
cv2.destroyAllWindows()'''


'''

## stepp three

model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)

def get_person(img, face_point):
    results = model(img)

    p = pd.DataFrame(results.xyxy[0].cpu().numpy())

    df = p[p[5] == 0]

    for i in range(len(df)):
        l = [int(x) for x in df.iloc[i].tolist()[:-2]]
        if l[0] <= face_point[0] <= l[2] and l[0] <= face_point[2] <= l[2]:
            if l[1] <= face_point[1] <= l[3] and l[1] <= face_point[3] <= l[3]:
                return l
            


list_name = [ 'train', 'test']

for n in list_name:           

    body_x_min = []
    body_y_min = []
    body_x_max = []
    body_y_max = []

    df = pd.read_csv(f'{n}.csv')
    df1 = df

    for i in range(len(df)):
        print(i)
        face_point = [int(x) for x in df.iloc[i].tolist()[3:-3]]

        try:
            img = Image.open('./imdb/'+df.iloc[i].tolist()[0])

            body = get_person(img, face_point)

            if body == None:
                # img = cv2.imread('./imdb/'+df.iloc[i].tolist()[0])

                # reg = cv2.rectangle(img, (face_point[0], face_point[1]), (face_point[2], face_point[3]), (0, 0, 0), 2)

                # cv2.imshow('fr2ame', reg)
                # cv2.waitKey(500)
                # cv2.destroyAllWindows()
                print('none')

                df1 = df1.drop(index=i)
            else:
            
                body_x_min.append(body[0])
                body_y_min.append(body[1])
                body_x_max.append(body[2])
                body_y_max.append(body[3])
            
        except:
            print('none')

            df1 = df1.drop(index=i)




    df1['body_x_min'] = body_x_min
    df1['body_y_min'] = body_y_min
    df1['body_x_max'] = body_x_max
    df1['body_y_max'] = body_y_max


    df1.to_csv(f'{n}1.csv', index=False)'''

# df = pd.read_csv('./part_of_IMDB(body+face).csv')
# d = IMDBDataset(df, T.Compose([T.Resize((224, 224)),T.ToTensor(),]),  T.Compose([T.Resize((224, 224)),T.ToTensor(),])).__getitem__(89)

# print(d[0])


crop_body_face_csv('./imdb_part')

