from functools import reduce
from torch import nn
from torch.nn import functional as F

class LeNet(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(64, 32)
       self.fc2 = nn.Linear(32, 10)

   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       x = self.fc2(x)
       return x

   def num_flat_features(self, features):
       return reduce(lambda x, y: x * y, features.shape[1:])