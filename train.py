import torch
import argparse
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from datasets import CatDogDataset
from model import VGG16Feature


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
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    workers = config['train']['workers']
    print_freq = config['train']['print_freq']
    learning_rate = config['train']['learning_rate']
    momentum = config['train']['momentum']
    weight_decay = config['train']['weight_decay']
    train_dataset_csv = config['train']['train_dataset_csv']
    val_dataset_csv = config['train']['val_dataset_csv']
    test_interval = config['train']['test_interval'] # Number of epoches between testing phases
    early_stopping_param = config['train']['early_stopping_param']
    early_stopping_patience = config['train']['early_stopping_patience']

    epoch_since_improvement = 0
    best_loss = 1e10
    best_epoch = 0

    # Load dataset
    training_set = CatDogDataset(file_csv = train_dataset_csv, classes = classes)
    training_loader = DataLoader(training_set,
                                 batch_size = batch_size,
                                 shuffle = True,
                                 drop_last = True,
                                 num_workers = workers)                        

    testing_set = CatDogDataset(file_csv = val_dataset_csv, classes = classes)
    testing_loader = DataLoader(testing_set,
                                batch_size = batch_size,
                                shuffle = False,
                                drop_last = False,
                                num_workers = workers)


    # Load VGG16 model 
    vgg16 = VGG16Feature()
    vgg16.to(dev)
    
    # Loss 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg16.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
    
    

    # Train now
    vgg16.train()
    num_iter_per_epoch = len(training_loader)
    for epoch in range(epochs):
        for i, batch in enumerate(training_loader, 0):
            images, labels = batch
            '''
            if torch.cuda.is_available():
                images = Variable(images.cuda(), requires_grad = True)
            else:
                images = Variable(images, requires_grad = True)
            '''
            images = images.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            # Forward
            outputs = vgg16(images)
            #print(images)

            #print(outputs)
            
            loss = criterion(outputs, labels)
            print(loss)
        
            # Backward 
            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f}".format(
                epoch + 1,
                epochs,
                iter + 1,
                num_iter_per_epoch,
                learning_rate,
                loss
                ))

        if epoch % test_interval == 0:
            vgg16.eval()
            loss_ls = []
            for i, batch in enumerate(testing_loader, 0):
                images, labels = batch
                num_label = len(labels)
                '''
                if torch.cuda.is_available():
                    images = images.cuda()
                '''
                images = images.to(dev)
                labels = labels.to(dev)

                with torch.no_grad():
                    outputs = vgg16(images)
                    loss = criterion(outputs, labels)
                loss_ls.append(loss * num_label)
            avg_loss = sum(loss_ls)/ testing_set.__len__()
            
            print("Epoch: {}/{}, Lr: {}, Loss:{:.2f}".format(
                epoch + 1,
                epochs,
                learning_rate,
                avg_loss
                ))

            if avg_loss + early_stopping_param < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                torch.save(vgg16, "model/" + "whole_trained_vgg16_model.pt")
                torch.save(vgg16.state_dict(), "model/" + "state_dict_trained_vgg16_model.pt")

            # Early stopping
            if epoch - best_epoch > early_stopping_patience:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, avg_loss))
                break
        

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)                                                            