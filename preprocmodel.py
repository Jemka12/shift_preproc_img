from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm, tqdm_notebook

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

import numpy as np
import os
import shutil
from pathlib import Path
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda')

class FlipDataset(Dataset):
  def __init__(self, files):
    super().__init__()
    self.files = files
    self.len_ = len(self.files)

  def __len__(self):
    return self.len_
  
  def load_sample(self, file):
    image = Image.open(file)
    image.load()
    return image

  def __getitem__(self, index):
    transform = transforms.Compose([
      transforms.ToPILImage(mode="RGB"),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    x = self.load_sample(self.files[index])
    x = self._prepare_sample(x)
    x = transform(x)
    return x
  
  def _prepare_sample(self, image):
    image = image.resize((256, 256))
    return np.array(image)

class FlipImage():
  def __init__(self):
    self.model = self.model_push()
    self.label_encoder = LabelEncoder().fit(['flip', 'yes_flip_-90','yes_flip_90'])
  
  def model_push(self, path_model='/content/resnet_50_3_flip_0d987730.pt'):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 3)
    model.load_state_dict(torch.load(path_model))
    return model

  def predict(self, model, test_loader):
    with torch.no_grad():
      logits = []

      for inputs in test_loader:
        inputs = inputs.to(DEVICE)
        model.eval()
        outputs = model(inputs).cpu()
        logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs

  def flip_predict(self, model, image_path):
    test_dataset = FlipDataset(image_path)

    model.eval()
    imgs = [test_dataset[id].unsqueeze(0) for id in range(len(image_path))]
    probs_imgs = self.predict(model.cuda(), imgs)
    preds_class = np.argmax(probs_imgs, -1)

    preds_class = self.label_encoder.inverse_transform(preds_class)

    return preds_class
  
  def flip_image(self, test_file, preds_class, new_dir):
    try:
      os.mkdir(new_dir)
    except FileExistsError:
      pass
    transform_90 = transforms.Compose([
    transforms.ToPILImage(mode="RGB"),
    transforms.RandomRotation((-90,-90)),])

    transform_neg90 = transforms.Compose([
    transforms.ToPILImage(mode="RGB"),
    transforms.RandomRotation((90,90)),])
    
    for path_image, model_class in tqdm_notebook(zip(test_file, preds_class)):
      image = Image.open(str(path_image))
      image = np.array(image)
      if model_class == 'yes_flip_90' or  model_class == 'yes_flip_-90':
        new_image = transform_90(image)
        new_image.save(new_dir+path_image.name)
      elif model_class == 'flip':
        shutil.copyfile(path_image, new_dir+path_image.name)
    
  def flip_main(self, DIR, new_dir='Flip_models/'):
      print('Помните! Функция не работает с вложенными папками.')
      DIR = Path(DIR)
      image_file = list(sorted(DIR.rglob('*.*')))
      print('Определяю классы')
      predict = self.flip_predict(self.model, image_file)

      plt.figure(figsize=(12,10))
      sns.countplot(predict)
      print('Закончил')
      return image_file, predict

class DublicateData(Dataset):
  def __init__(self, files):
    super().__init__()
    self.files = files
    self.len_ = len(self.files)

  def __len__(self):
    return self.len_
  
  def load_sample(self, file):
    image = Image.open(file)
    image.load()
    return image

  def __getitem__(self, index):
    transform = transforms.Compose([
      transforms.ToPILImage(mode="RGB"),
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
    ])
    x = self.load_sample(self.files[index])
    x = self._prepare_sample(x)

    x = transform(x)
    y = self.files[index]
    return x, index
  
  def _prepare_sample(self, image):
    return np.array(image)

class DublicateImage():
  def __init__(self):
    super().__init__()
    self.model = self.model_push()
  
  def model_push(self):
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.cuda()
    return model

  def model_fit(self, model, loader, image_file):
    model.eval()
    outputs = torch.Tensor().cpu()
    index_list = []
    for inputs, idx in loader:
        inputs = inputs.to(DEVICE)
        out = model(inputs).squeeze().detach().cpu()
        
        outputs = torch.cat((outputs, out), 0)
        index_list += [image_file[i] for i in idx]
    return outputs, index_list

  def search(self, files, model,  batch_size, border, image_file):
    cosine = torch.nn.CosineSimilarity()
    loader = DataLoader(files, batch_size=batch_size, shuffle=False)
    outputs, path = self.model_fit(model, loader, image_file)
    outputs = list(zip(outputs, path))
    out = []
    print('Проверка на дубликаты')
    for first_element in tqdm(range(len(outputs))):
      x = first_element
      while x < len(outputs)-1:
        x += 1
        if cosine(outputs[first_element][0].unsqueeze(0), outputs[x][0].unsqueeze(0)) > border:
          out.append((outputs[first_element][1], outputs[x][1])) # путь до (оригинала, дубликата)
          #out.append(outputs[second_element][1]) - добавляет один путь до дубликата
          del outputs[x]
          x -= 1
    return out

  def dublicate_find(self, DIR, bath_size=120, border=0.97):
    DIR = Path(DIR)
    image_file = list(sorted(DIR.rglob('*.jpg')))
    data = DublicateData(image_file)
    print('Начинаю поиск')
    predict = self.search(data, self.model, bath_size, border, image_file)
    print(f'Поиск закончен. Найдено {len(predict)} файлов')
    return predict

class TarifClass():
  def __init__(self):
    super().__init__()
    self.model = self.model_push()
    self.label_encoder = LabelEncoder().fit(['m_tarif', 'o_tarif'])

  def model_push(self, path='/content/resnet_50_o_m_model_9774066797642436.pt'):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 2)) 
    model.load_state_dict(torch.load(path))
    model.cuda()
    return model

  def predict(self, model, test_loader):
    with torch.no_grad():
      logits = []

      for inputs in test_loader:
        inputs = inputs.to(DEVICE)
        model.eval()
        outputs = model(inputs).cpu()
        logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs
  
  def tarif_predict(self, model, image_path):
    test_dataset = FlipDataset(image_path)

    model.eval()
    imgs = [test_dataset[id].unsqueeze(0) for id in range(len(image_path))]
    probs_imgs = self.predict(model.cuda(), imgs)
    preds_class = np.argmax(probs_imgs, -1)

    preds_class = self.label_encoder.inverse_transform(preds_class)

    return preds_class
  
  def tarif_main(self, DIR):
    DIR = Path(DIR)
    image_file = list(sorted(DIR.rglob('*.*')))
    print('Определяю классы')
    predict = self.tarif_predict(self.model, image_file)

    plt.figure(figsize=(12,10))
    sns.countplot(predict)
    print('Закончил')
    return image_file, predict

def load_model():
  os.system('gdown https://drive.google.com/uc?id=18lhZKdp1wrHcJmeidTqdsv_zSF1aQ54R&export=download')
  os.system('gdown https://drive.google.com/uc?id=1OQ9lYBwkXQUmT9ipw_ca1CaFuH1sBgg5&export=download')
  print('Модели скачаны')
