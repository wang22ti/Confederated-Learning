from dataset import generate_noise, get_office
from utils import cal_ece

import numpy as np
import random

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
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
    num_epoch = 50
    lr, momentum, weight_decay = 0.1, 0.9, 1e-4
    num_classes = 31
    noise_rate = 0.4
    source = ['amazon', 'webcam']
    target_name = 'dslr'
    for alpha in [1.2, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        init(0)

        # prepare data
        print('reweighting, target:', target_name, 'noise rate:', noise_rate, 'alpha:', alpha, 'lr:', lr)
        dataset = get_office(target_name)
        dataset_size = len(dataset)
        num_train, num_val = int(0.8 * dataset_size), int(0.1 * dataset_size)
        num_test = dataset_size - (num_train + num_val)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])
        dataset = generate_noise(dataset, train_dataset.indices, num_noise=int(num_train * noise_rate))
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # train
        model = resnet18(pretrained=False)
        model.fc = Linear(512, num_classes, bias=True)
        model = model.cuda()
        source_model = torch.load('checkpoints/{}_best.pkl'.format('-'.join(source))).cuda()

        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        best_val_acc = 0
        for epoch in range(num_epoch):
            correct = 0
            for data, target in train_loader:
                data, target = data.cuda(), target.long().cuda()      

                with torch.no_grad():
                    source_pred = torch.softmax(source_model(data), dim=1)
                    raw_target = torch.zeros(data.shape[0], 31).cuda().scatter_(1, target.unsqueeze(dim=-1).long(), 1)
                    weight = torch.exp(-torch.norm(raw_target - source_pred, dim=1) * alpha)
                    
                pred = model(data)
                pred_labels = torch.argmax(pred, axis=1)
                correct += torch.sum((pred_labels == target).int(), axis=-1)
                loss = CrossEntropyLoss(reduction='none')(pred, target) * weight
                
                optimizer.zero_grad()
                loss.mean().backward()
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
                torch.save(model, 'checkpoints/{}_{}_reweighting_best.pkl'.format(target_name, noise_rate))
                best_val_acc = val_acc
        print('best val acc:', best_val_acc)

        # test
        model = torch.load('checkpoints/{}_{}_reweighting_best.pkl'.format(target_name, noise_rate))
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
        print()