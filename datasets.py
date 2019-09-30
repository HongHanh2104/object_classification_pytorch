import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class CatDogDataset(Dataset):
    def __init__(self, file_csv, classes, image_size = 224, is_training = True):
        full_data = pd.read_csv(file_csv)
        _id = full_data['id']
        _label = full_data['label']
        self.id = np.array(_id)
        self.label = np.array(_label)
        self.classes = classes
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.is_training = is_training
    
    def __getitem__(self, index):
        _id = self.id[index]
        image_path = os.path.join("database/images/", "{}.jpg".format(_id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(self.label[index], dtype = np.long)

    def __len__(self):
        return len(self.id)


#classes = ['dog', 'cat']
#VOCDataset(file_csv = 'database/training/train_labels.csv', classes = classes).__getitem__(1)