from dataset import get_digits
from model import LeNet

import numpy as np
import random

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader


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
    num_epoch = 200
    lr, momentum, weight_decay = 0.01, 0.9, 1e-4
    source = 'mnist'
    init(0)

    print(source)
    # prepare data
    train_dataset, test_dataset = get_digits(source)
    num_train, num_test = len(train_dataset), len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    
    # train
    model = LeNet().cuda()
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
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            pred = model(data).detach()
            pred_labels = torch.argmax(pred, axis=1)
            correct += torch.sum((pred_labels == target).int(), axis=-1)

        val_acc = int(correct) / num_test
        print('epoch {}, train_acc: {:.4f}, val_acc: {:.4f}'.format(epoch, train_acc, val_acc))
        if val_acc > best_val_acc:
            torch.save(model, 'checkpoints/{}_best.pkl'.format(source))
            best_val_acc = val_acc
    print('best val acc:', best_val_acc)