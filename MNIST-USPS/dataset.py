import random
import numpy as np

from torchvision.datasets import MNIST, USPS
import torchvision.transforms as transforms

def generate_noise(dataset, indices, num_noise=0, num_classes=10):
    random.shuffle(indices)
    
    noise_indx = indices[: num_noise]
    noise_label = np.random.randint(num_classes, size=num_noise)
    for idx, label in zip(noise_indx, noise_label):
        dataset.targets[idx] = label
    return dataset

def get_noise_idx(dataset, indices, num_noise=0, num_classes=10):
    random.shuffle(indices)
    
    noise_indx = indices[: num_noise]
    noise_label = np.random.randint(num_classes, size=num_noise)
    clean_label = list()
    for idx, label in zip(noise_indx, noise_label):
        clean_label.append(dataset.imgs[idx][1])
        dataset.imgs[idx] = (dataset.imgs[idx][0], label)
    return dataset, noise_indx, noise_label, clean_label


def get_digits(name_dataset, download=False):
    if name_dataset == 'mnist':
        data_transforms = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ])
        train_set = MNIST('dataset', train=True, transform=data_transforms, download=download)
        test_set = MNIST('dataset', train=False, transform=data_transforms, download=download)
    elif name_dataset == 'usps':
        train_set = USPS('dataset', train=True, transform=transforms.ToTensor(), download=download)
        test_set = USPS('dataset', train=False, transform=transforms.ToTensor(), download=download)

    return train_set, test_set