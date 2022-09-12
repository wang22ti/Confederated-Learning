from dataset import get_office
from utils import cal_ece

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, random_split

def init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    batch_size = 256
    source = ['dslr', 'webcam']
    target_name = 'amazon'
    init(0)

    # prepare data
    dataset = get_office(target_name)
    dataset_size = len(dataset)
    num_train, num_val = int(0.8 * dataset_size), int(0.1 * dataset_size)
    num_test = dataset_size - (num_train + num_val)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # test
    model = torch.load('checkpoints/{}_best.pkl'.format('-'.join(source)))
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