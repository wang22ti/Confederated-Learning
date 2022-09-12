from dataset import generate_noise, get_digits
from utils import cal_ece
from model import LeNet

import numpy as np
import random

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

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
    noise_rate = 0.8
    source = 'mnist'
    target_name = 'usps'

    log_file = open('log/{}_{}_reg.log'.format(target_name, noise_rate), 'a')
    for alpha in [10**_ for _ in range(-4, -9, -1)]:
        init(0)

        # prepare data
        print('reg, target:', target_name, 'noise rate:', noise_rate, 'alpha:', alpha)
        print('reg, target:', target_name, 'noise rate:', noise_rate, 'alpha:', alpha, file=log_file)
        dataset, test_dataset = get_digits(target_name)
        dataset_size = len(dataset)
        num_train  = 1000
        num_val = dataset_size - num_train
        num_test = len(test_dataset)

        train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
        dataset = generate_noise(dataset, train_dataset.indices, num_noise=int(num_train * noise_rate))
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # train
        model = LeNet().cuda()
        source_model = torch.load('checkpoints/{}_best.pkl'.format(source)).cuda()

        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        best_val_acc = 0
        for epoch in range(num_epoch):
            correct = 0
            for data, target in train_loader:
                data, target = data.cuda(), target.long().cuda()      

                with torch.no_grad():
                    source_pred = torch.softmax(source_model(data), dim=1)
                    
                pred = model(data)
                pred_labels = torch.argmax(pred, axis=1)
                correct += torch.sum((pred_labels == target).int(), axis=-1)

                reg = torch.mean(torch.norm(torch.softmax(pred, dim=1) - source_pred, dim=1))
                loss = CrossEntropyLoss()(pred, target) + alpha * reg
                
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
            print('epoch {}, train_acc: {:.4f}, val_acc: {:.4f}'.format(epoch, train_acc, val_acc), file=log_file)
            if val_acc > best_val_acc:
                torch.save(model, 'checkpoints/{}_{}_reg_best.pkl'.format(target_name, noise_rate))
                best_val_acc = val_acc
        print('best val acc:', best_val_acc, file=log_file)

        # test
        model = torch.load('checkpoints/{}_{}_reg_best.pkl'.format(target_name, noise_rate))
        correct = 0
        py_list, y_index_list = list(), list()
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            pred = model(data).detach()
            pred_labels = torch.argmax(pred, axis=1)
            correct += torch.sum((pred_labels == target).int(), axis=-1)

            py_list.append(pred)
            y_index_list.append(target)

        acc = int(correct) / num_test
        py, y_index = torch.softmax(torch.cat(py_list, dim=0), dim=1).cpu().numpy(), torch.cat(y_index_list, dim=0).cpu().numpy()
        ece = cal_ece(py, y_index, n_bins=15)
        print('test acc:', acc, 'test ecc:', ece)
        print('test acc:', acc, 'test ecc:', ece, file=log_file)
        print(file=log_file)