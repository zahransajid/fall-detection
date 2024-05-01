import torch
from torch import nn
from torchsummary import summary

class Fall_model(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters, pool_size):
        super(Fall_model, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size)

        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size)

        self.conv3 = nn.Conv1d(
            num_filters * 2, num_filters * 4, kernel_size=kernel_size
        )

        self.fc1 = nn.Linear(int(num_filters * 4)*370, int(num_filters//2)*370)
        self.relul = nn.ReLU()
        self.fc2 = nn.Linear(int(num_filters//2)*370, 3)

    def forward(self, x):
        
        x = self.relul(self.conv1(x))
        x = self.pool(x)
        x = self.relul(self.conv2(x))
        x = self.relul(self.conv3(x))
        # x is of shape batch_size*num_channels(8)*3000
        x = x.view(x.size(0), x.size(1)*x.size(2))
        x = self.relul(self.fc1(x))

        output = nn.functional.softmax(self.fc2(x), dim=1)
        return output

if __name__ == '__main__':
    model = Fall_model(in_channels=8,kernel_size=(3),num_filters=16,pool_size=8)
    summary(model,(8,3000),device='cpu')