import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def logging_setting(run_name):
    if not os.path.exists("model"):
        os.mkdir("model")
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(os.path.join("model", run_name)):
        os.mkdir(os.path.join("model", run_name))
    if not os.path.exists(os.path.join("results", run_name)):
        os.mkdir(os.path.join("results", run_name))


# dataset: MNIST_fashion
def get_data(batch_size, img_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5)),

    ])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

