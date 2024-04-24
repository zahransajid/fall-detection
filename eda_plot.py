from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    df1 = pd.read_excel(r"SFU-IMU Dataset\IMU Dataset\sub1\Near_Falls\AXR_slip_trial4.xlsx")
    df2 = pd.read_excel(r"SFU-IMU Dataset\IMU Dataset\sub1\Falls\AXR_ITCS_trial4.xlsx")
    df3 = pd.read_excel(r"SFU-IMU Dataset\IMU Dataset\sub1\ADLs\AXR_DS_trial2.xlsx")
    used_columns = []
    for c in df1.columns:
        if "sternum" in c and "m/s" in c: 
            used_columns.append(df1[c])
    used_columns = np.array(used_columns,dtype=np.float32)
    abs_vel_1 = np.linalg.norm(used_columns,ord=2,axis=0)
    used_columns = []
    for c in df2.columns:
        if "sternum" in c and "m/s" in c: 
            used_columns.append(df2[c])
    used_columns = np.array(used_columns,dtype=np.float32)
    abs_vel_2 = np.linalg.norm(used_columns,ord=2,axis=0)
    used_columns = []
    for c in df3.columns:
        if "sternum" in c and "m/s" in c: 
            used_columns.append(df3[c])
    used_columns = np.array(used_columns,dtype=np.float32)
    abs_vel_3 = np.linalg.norm(used_columns,ord=2,axis=0)
    plt.subplot(3,1,1)
    plt.plot(list(range(len(abs_vel_1))),abs_vel_1)
    plt.title("Near fall")
    plt.subplot(3,1,2)
    plt.plot(list(range(len(abs_vel_2))),abs_vel_2)
    plt.title("Fall")
    plt.subplot(3,1,3)
    plt.plot(list(range(len(abs_vel_3))),abs_vel_3)
    plt.title("Normal data")
    plt.show()