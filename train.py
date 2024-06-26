import torch
import torch.nn as nn
import torch.optim as optim
from models.CNN1D import Fall_model
from dataset import IMUDataset
from torch.utils.data import DataLoader
import numpy as np

label_transform = {
    "Near_Falls": [0.0, 0.0, 1.0],
    "Falls": [0.0, 1.0, 0.0],
    "ADLs": [1.0, 0.0, 0.0]
}
device = torch.device('cuda:0')
model = Fall_model(in_channels=8,kernel_size=(3),num_filters=4,pool_size=8).to(device)

num_epochs = 10
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
with open("file_list.txt") as f:
    data_paths = f.read().rstrip().split("\n")
train_loader = DataLoader(IMUDataset(data_paths), batch_size=32,shuffle=True)

for epoch in range(num_epochs):
    for batch_idx, (targets, inputs) in enumerate(train_loader):
        optimizer.zero_grad()
        targets = torch.Tensor([label_transform[x] for x in targets]).to(device)
        inputs = inputs.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if batch_idx % 3 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    if(epoch%2 == 0):
        path = "./weights/model{epoch}.pth".format(
            epoch=epoch+1)
        torch.save(model.state_dict(), path)

print("Training complete!")