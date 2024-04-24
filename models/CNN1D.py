import torch
from torch import nn
class Fall_model(nn.Module):
  def _init_(self, in_channels, kernel_size, num_filters, pool_size):
    super(Fall_model, self)._init_()

    self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size)

    self.pool = nn.MaxPool1d(kernel_size=pool_size)
    
    self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size)
 
    self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=kernel_size)

    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(num_filters * 4, int(num_filters * 2))
    self.relul = nn.ReLU()
    self.fc2 = nn.Linear(int(num_filters * 2), 3)  

  def forward(self, x):
  
    x = self.relul(self.conv1(x))
    x = self.pool(x)
    x = self.relul(self.conv2(x))
    x = self.relul(self.conv3(x))

    x = self.flatten(x)
    x = self.relul(self.fc1(x))

    output = nn.functional.softmax(self.fc2(x), dim=1)
    return output


model = Fall_model(a, b, c, d)  
for data, label in data_loader:

  output = model(data)