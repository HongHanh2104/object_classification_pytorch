import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from datasets import CatDogDataset
from model import VGG16Feature
from torch.utils.data.sampler import SubsetRandomSampler

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    dev = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    classes = config['model']['labels']
    test_dataset_csv = config['test']['test_dataset_csv']
    trained_model = config['test']['trained_model']

    test_transforms = transforms.Compose([transforms.ToTensor()])

    # Load VGG16 trained model 
    vgg16 = torch.load(trained_model, map_location = dev)
    vgg16.to(dev)
    
    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(test_dataset_csv, classes, 5)
    
    fig = plt.figure(figsize=(10,10))
    for i in range(len(images)):
        image = to_pil(images[i])
        image_tensor = images[i]
        label = predict_image(vgg16, image_tensor)
        sub = fig.add_subplot(1, len(images), i + 1)
        res = int(labels[i]) == label
        sub.set_title(str(classes[label]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()

    
def predict_image(model, image):
    image = image.unsqueeze_(0)
    input = Variable(image)
    output = model(input)
    label = output.data.numpy().argmax()
    
    return label


def get_random_images(test_dataset_csv, classes, number):
    # Load testing dataset
    testing_set = CatDogDataset(file_csv = test_dataset_csv, classes = classes)
    indices = list(range(len(testing_set)))
    np.random.shuffle(indices)
    idx = indices[:number]
    sampler = SubsetRandomSampler(idx)
    
    testing_loader = DataLoader(testing_set,
                                sampler = sampler,
                                batch_size = number)

    dataiter = iter(testing_loader)
    images, labels = dataiter.next()
    return images, labels
   

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)                                                            
