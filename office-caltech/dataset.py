import os
import random
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

def generate_noise(dataset, indices, num_noise=0, num_classes=10):
    random.shuffle(indices)
    
    noise_indx = indices[: num_noise]
    noise_label = np.random.randint(num_classes, size=num_noise)
    for idx, label in zip(noise_indx, noise_label):
        dataset.imgs[idx] = (dataset.imgs[idx][0], label)
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

def get_office(name_dataset):
    root_dir = os.path.join('office_caltech', name_dataset)

    # https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/data_loader.py
    mean_std = {
        "amazon":{
            "mean":[0.7923, 0.7862, 0.7841],
            "std":[0.3149, 0.3174, 0.3193]
        },
        "dslr":{
            "mean":[0.4708, 0.4486, 0.4063],
            "std":[0.2039, 0.1920, 0.1996]
        },
        "webcam":{
            "mean":[0.6119, 0.6187, 0.6173],
            "std":[0.2506, 0.2555, 0.2577]
        },
        "caltech": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }

    }

    data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[name_dataset]["mean"],
                                 std=mean_std[name_dataset]["std"])
        ])

    dataset = datasets.ImageFolder(root=root_dir, transform=data_transforms)

    return dataset