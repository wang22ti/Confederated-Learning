from dataset import get_office

import numpy as np
import random

import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.models import resnet18

def init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.set_default_tensor_type(torch.cuda.DoubleTensor)

if __name__ == '__main__':
    batch_size = 256
    num_epoch = 100
    lr, momentum, weight_decay = 0.1, 0.9, 1e-4
    num_classes = 31
    source = ['dslr', 'webcam']
    init(0)

    print(source)
    # prepare data
    dataset = ConcatDataset([get_office(s) for s in source])
    dataset_size = len(dataset)
    num_train = int(0.9 * dataset_size)
    num_val = dataset_size - num_train 
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # train
    model = resnet18(pretrained=False)
    model.fc = Linear(512, num_classes, bias=True)
    model = model.cuda()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    best_val_acc = 0
    for epoch in range(num_epoch):
        correct = 0
        for data, target in train_loader:
            data, target = data.cuda(), target.long().cuda()            
            pred = model(data)
            pred_labels = torch.argmax(pred, axis=1)
            correct += torch.sum((pred_labels == target).int(), axis=-1)
            loss = CrossEntropyLoss()(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = int(correct) / num_train
        
        correct = 0
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            pred = model(data).detach()
            pred_labels = torch.argmax(pred, axis=1)
            correct += torch.sum((pred_labels == target).int(), axis=-1)

        val_acc = int(correct) / num_val
        print('epoch {}, train_acc: {:.4f}, val_acc: {:.4f}'.format(epoch, train_acc, val_acc))
        if val_acc > best_val_acc:
            torch.save(model, 'checkpoints/{}_best.pkl'.format('-'.join(source)))
            best_val_acc = val_acc
    print('best val acc:', best_val_acc)