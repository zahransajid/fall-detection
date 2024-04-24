from typing import List
import torch
import os
import pandas as pd
import numpy as np
from preprocess import Preprocessor


class IMUDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths: List[str], generate_labels : bool =True):
        self.data_paths = data_paths
        self.generate_labels = generate_labels
        for path in self.data_paths:
            assert (path.endswith(".xlsx"))
        self.PAD_LENGTH = 3000
        self._len = len(self.data_paths)
        self.preprocessor = Preprocessor()
        self.TO_EXTRACT = [
            [
                "sternum Acceleration X (m/s^2)",
                "sternum Acceleration Y (m/s^2)",
                "sternum Acceleration Z (m/s^2)",
            ],
            [
                "sternum Angular Velocity X (rad/s)",
                "sternum Angular Velocity Y (rad/s)",
                "sternum Angular Velocity Z (rad/s)",
            ],
            [
                "waist Acceleration X (m/s^2)",
                "waist Acceleration Y (m/s^2)",
                "waist Acceleration Z (m/s^2)",
            ],
            [
                "waist Angular Velocity X (rad/s)",
                "waist Angular Velocity Y (rad/s)",
                "waist Angular Velocity Z (rad/s)",
            ],
            [
                "r.thigh Acceleration X (m/s^2)",
                "r.thigh Acceleration Y (m/s^2)",
                "r.thigh Acceleration Z (m/s^2)",
            ],
            [
                "r.thigh Angular Velocity X (rad/s)",
                "r.thigh Angular Velocity Y (rad/s)",
                "r.thigh Angular Velocity Z (rad/s)",
            ],
            [
                "l.thigh Acceleration X (m/s^2)",
                "l.thigh Acceleration Y (m/s^2)",
                "r.thigh Acceleration Z (m/s^2)",
            ],
            [
                "l.thigh Angular Velocity X (rad/s)",
                "l.thigh Angular Velocity Y (rad/s)",
                "l.thigh Angular Velocity Z (rad/s)",
            ],
        ]
        self.N_CHANNELS = len(self.TO_EXTRACT)
    def __len__(self):
        return self._len
    def __getitem__(self, index : int):
        path = self.data_paths[index]
        df = pd.read_excel(path)
        ret = []
        for channel in self.TO_EXTRACT:
            arr = df[channel].to_numpy()
            arr = np.linalg.norm(arr,ord=2,axis=1)
            arr = self.preprocessor.process(arr)
            if(len(arr) > self.PAD_LENGTH):
                arr = arr[:self.PAD_LENGTH]
            arr = arr.tolist() + [0]*(self.PAD_LENGTH - len(arr))
            ret.append(arr)
        if(self.generate_labels):
            labels = {
                "Near_Falls" : "Near_Falls" in path,
                "Falls" : "Falls" in path,
                "ADLs": "ADLs" in path
            }
            for l in labels.keys():
                if(labels[l]):
                    return (l,torch.tensor(ret))
            raise Exception("Could not find label from path")
        else:
            return torch.tensor(ret)

if __name__ == '__main__':
    dl = IMUDataset([
        r"SFU-IMU Dataset\IMU Dataset\sub2\ADLs\TXI_DSS_trial1.xlsx",
    ])
    for x in dl:
        print(x)