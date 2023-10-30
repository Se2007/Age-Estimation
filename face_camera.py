import cv2
import torch
import model
from torchvision import transforms as T
from facenet_pytorch import MTCNN
from face_detection import RetinaFace
from PIL import Image


vid = cv2.VideoCapture(0)

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


RGB_MEAN = [ 0.485, 0.456, 0.406 ]
RGB_STD = [ 0.229, 0.224, 0.225 ]
load_path = './model/' + 'model_loss0.5932' + ".pth"

model = model.Resnet(120, reset=False)
model = load(model, device='cpu', load_path = load_path)

transform = T.Compose([
                T.Resize((128,128)), 
                T.ToTensor(),
                T.Normalize(mean = RGB_MEAN, std = RGB_STD),
            ])
detector = RetinaFace()

model.eval()
  
while vid.isOpened():
      
    ret, frame = vid.read()

    faces = detector(frame)
    box, landmarks, score = faces[0]
    x, y, w, h = map(int, (box[0], box[1], box[2], box[3]))
                    

    # mtcnn = MTCNN()

    # boxes, _ = mtcnn.detect(frame)
    # if boxes is not None :
    #     for x,y,w,h in boxes:
    #         x, y, w, h = map(int, (x, y, w, h))
    #                         # x, y, w, h = abs(x), abs(y), abs(w), abs(h)
    #         x, y, w, h = map(abs, (x, y, w, h))

    cut = frame[y:h, x:w]
    # cut = cv2.cvtColor(cut, cv2.COLOR_RGB2BGR)

    im = Image.fromarray(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))
    im = transform(im)

    age_pred = model(im.unsqueeze(0))

    # age_pred = model(transform(torch.FloatTensor(cut).permute(-1, 1, 0)).unsqueeze(0))

    # print(age_pred.max(1)[1])#[1]
    print(age_pred.argmax())
    reg = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 0), 2)

    # for (x, y, w, h) in face:
    #     cut = frame[y:y+h, x:x+w]
    #     print(x,y)
    #     reg = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #     age_pred = model(transform(torch.FloatTensor(cut).permute(-1, 1, 0)).unsqueeze(0))

    #     print(age_pred.max(1)[1])
  
    cv2.imshow('frame', reg)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()