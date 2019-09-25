import torch
import utils
from torch.autograd import Variable
import torch.nn as nn
from model import VGG16Feature

def train():
    learning_rate = 0.001

    # Load VGG16 model 
    vgg16 = VGG16Feature()

    train_loader = utils.get_train_loader(batch_size = 4)

    # Loss 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg16.parameters(), lr = learning_rate)

    for epoch in range(5):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
        
        optimizer.zero_grad()
        y_pred = vgg16(images)

        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print("Epoch [%d/ %d], Iter [%d/ %d] Loss: %.4f" % (epoch, 5, i + 1, 200, running_loss)) 

    
    print('Finished Training.')
    torch.save(vgg16.state_dict(), './vgg16.pt')
    print('Saved model parameters to disk.')

train()