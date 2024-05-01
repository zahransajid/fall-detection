
import json
from random import shuffle
import torch
from models.CNN1D import Fall_model
from torch.utils.data import DataLoader
from dataset import IMUDataset
import os
import numpy as np
from sklearn import metrics

device = torch.device('cuda')
EPOCH=9
labels = {
    0 : "ADLs",
    1 : "Falls",
    2 : "Near_Falls",
}
labels_inv = {
    "ADLs" : 0,
    "Falls" : 1,
    "Near_Falls" : 2,
}
def load_model():
    model = Fall_model(in_channels=8,kernel_size=(3),num_filters=4,pool_size=8).to(device)
    path = f"./weights/model{EPOCH}.pth"
    
    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model


if __name__ == '__main__':
    model = load_model()
    with open('file_list.txt') as f:
        paths = f.read().rstrip().split("\n")
        shuffle(paths)
        paths = paths
    
    dataset = IMUDataset(paths)
    loader = DataLoader(dataset,batch_size=1,shuffle=False)
    y_test = []
    y_pred = []
    N = len(paths)
    with torch.no_grad():
        for i, (label, t) in enumerate(loader):
            t = t.to(device)
            results = model(t)
            results = results.to('cpu').numpy()
            computed_label = labels[np.argmax(results)]
            
            label = label[0]
            print(f"{i}/{N}: ({label} : {computed_label})")
            y_pred.append(int(np.argmax(results)))
            y_test.append(labels_inv[label])
    with open("results.json","w") as f:
        json.dump(
            {
                "y_test" : y_test,
                "y_pred" : y_pred,
            },
            f
        )