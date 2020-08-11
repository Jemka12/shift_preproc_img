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

class ClassModelData(Dataset):
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

class ManyClass():
  def __init__(self, class_model):
    super().__init__()
    self.label_encoder = LabelEncoder()
    self.class_model = class_model
    self.model = self.model_push()

  def model_push(self):
    if self.class_model == 'many':
      path='/content/resnet_50_Mtarif_9456342668863262.pt'
      last_layer = 11
      self.label_encoder.fit(['SOE', 'caskad', 'ce2726', 'energomera', 'heba', 'len_electro',
       'matrix', 'mercury_200.02', 'mercury_any', 'other', 'trash'])
    elif self.class_model == 'one':
      path='/content/resnet_50_Otarif_0.947729.pt'
      last_layer = 17
      self.label_encoder.fit(['SOLO', 'c0_446', 'co505', 'co_any', 'coe', 'ekf', 'energomera',
       'leneltktro', 'mercury_201', 'mercury_201.7', 'mercury_202.5',
       'mercury_any', 'neva_101_1s0', 'neva_103_1s0', 'other', 'seo',
       'trash'])

    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
      nn.Linear(2048, 1024),
      nn.ReLU(),
      nn.Dropout(),
      nn.BatchNorm1d(1024),
      nn.Linear(1024, last_layer)) 
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
    test_dataset = ClassModelData(image_path)

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