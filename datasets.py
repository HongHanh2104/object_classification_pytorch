import json
import os
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from utils import *

class VOCDataset(Dataset):
    def __init__(self, classes, image_size = 224, is_training = True):
        id_list_path = "database/training/Main/train.txt"
        self.ids = [id.strip() for id in open(id_list_path)]
        self.classes = classes
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __getitem__(self, index):
        id = self.ids[index]
        image_path = os.path.join("database/images", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_xml_path = os.path.join("database/annotations", "{}.xml".format(id))
        annot = ET.parse(image_xml_path)
        
        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in 
                                    ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])
        
        if self.is_training:
            transformation = Compose([VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])

        image, objects = transformation((image, objects))
        #return image, objects
        return np.transpose(np.array(image, dtype = np.float32), (2, 0, 1)), np.array(objects, dtype = np.float32)
        

    def __len__(self):
        return self.num_images

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
VOCDataset(classes = classes).__getitem__(1)
