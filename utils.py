import torch 
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def get_train_loader(batch_size):
    trainset = torchvision.datasets.CIFAR10(root = './CIFAR-10 Classifier Using CNN in PyTorch/data/',
                                        train = True,
                                        download = True,
                                        transform = transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size = batch_size,
                                            shuffle = True)
    return trainloader


def get_test_loader(batch_size):
    testset = torchvision.datasets.CIFAR10(root = './CIFAR-10 Classifier Using CNN in PyTorch/data',
                                        train = False,
                                        download = True,
                                        transform = transform)

    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size = batch_size,
                                            shuffle = False)
    return testloader

