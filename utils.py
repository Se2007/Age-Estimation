import torch
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch import nn
import torchmetrics as tm
from torchmetrics import Accuracy






class AverageMeter(object) :
  def __init__(self):
    self.reset()
  def reset(self) :
    self.avg = 0
    self.val = 0
    self.sum = 0
    self.count = 0
  def update (self, val, n=1) :
    self.val = val
    self.count += n
    self.sum += self.val * n
    self.avg = self.sum / self.count


def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=None, device='cpu'):
  
  metric = tm.MeanAbsoluteError().to(device)
  model.train().to(device)
  loss_train = AverageMeter()
  metric.reset()

  with tqdm(train_loader, unit='batch') as tepoch:
    for face, body, targets in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')
      torch.backends.cudnn.benchmark = False

      face = face.to(device)
      body = body.to(device)
      targets = targets.to(device)

      outputs =  model(face, body)#    model(face)

      loss = loss_fn(outputs, targets)

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item(), n=len(targets))
      metric.update(outputs, targets)

      tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())
      torch.backends.cudnn.benchmark = True

  return model, loss_train.avg, metric.compute().item()

def evaluate(model, test_loader, loss_fn, device='cpu'):
  metric = tm.MeanAbsoluteError().to(device)
  model.eval()
  loss_eval = AverageMeter()
  metric.reset()

  with torch.inference_mode():
    for face, body, targets in test_loader:
      face = face.to(device)
      body = body.to(device)
      targets = targets.to(device)

      outputs = model(face, body)  #   model(face)


      loss = loss_fn(outputs, targets)
      loss_eval.update(loss.item(), n=len(targets))

      metric(outputs, targets)

  return loss_eval.avg, metric.compute().item()


'''
def train(model, train_loader, optimizer, loss_fn, epoch=None, device='cpu'):
  model.train().to(device)
  trian_loss = AverageMeter()
  train_acc =  AverageMeter()  #  Accuracy(task='multiclass', num_classes=120).to(device)
  train_mae = AverageMeter()
  mae = nn.L1Loss()
  with tqdm(train_loader, unit='batch') as tepoch :
    for inputs, targets in tepoch:# utkface dataset need (inputs, targets, _)
      if epoch is not None:
        tepoch.set_description(f'Epoch {epoch}')
      inputs = inputs.to(device)
      targets = targets.to(device)
      
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      # print(inputs.shape[0])

      trian_loss.update(loss.item())
      _,perd = outputs.max(1)
      correct = perd.eq(targets).sum().item() / inputs.shape[0]
      train_mae.update(mae(perd.float(), targets.float()).item())
      train_acc.update(correct)
      #train_acc(outputs, targets)
      
      tepoch.set_postfix(loss=trian_loss.avg,
                         MAE=train_mae.avg)#train_acc.compute().item())
    return model, trian_loss.avg, train_acc.avg, train_mae.avg
  



def evaluet (model, test_loader, loss_fn, device='cpu'):
  model.eval()
  with torch.no_grad():
    test_acc = AverageMeter()
    test_loss = AverageMeter()
    test_mae = AverageMeter()
    mae = nn.L1Loss()
    for n, (inputs, targets) in enumerate(test_loader): # utkface dataset need (inputs, targets, _)
      inputs =  inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      loss = loss_fn(outputs, targets)

      test_loss.update(loss)
      _,perd = outputs.max(1)
      correct = perd.eq(targets).sum().item() / inputs.shape[0]
      test_mae.update(mae(perd.float(), targets.float()).item())
      test_acc.update(correct)
      return test_loss.avg, test_acc.avg, test_mae.avg
'''



