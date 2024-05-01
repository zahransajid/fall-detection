from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from preprocess import Preprocessor, inv_DFT_slow, DFT_slow


if __name__ == "__main__":
    p = Preprocessor()
    df1 = pd.read_excel(
        r"SFU-IMU Dataset\IMU Dataset\sub1\Near_Falls\AXR_slip_trial4.xlsx"
    )
    df2 = pd.read_excel(r"SFU-IMU Dataset\IMU Dataset\sub1\Falls\AXR_ITCS_trial4.xlsx")
    df3 = pd.read_excel(r"SFU-IMU Dataset\IMU Dataset\sub1\ADLs\AXR_DS_trial2.xlsx")
    used_columns = []
    for c in df1.columns:
        if "sternum" in c and "m/s" in c:
            used_columns.append(df1[c])
    used_columns = np.array(used_columns, dtype=np.float32)
    abs_vel_1 = np.linalg.norm(used_columns, ord=2, axis=0)
    used_columns = []
    for c in df2.columns:
        if "sternum" in c and "m/s" in c:
            used_columns.append(df2[c])
    used_columns = np.array(used_columns, dtype=np.float32)
    abs_vel_2 = np.linalg.norm(used_columns, ord=2, axis=0)
    used_columns = []
    for c in df3.columns:
        if "sternum" in c and "m/s" in c:
            used_columns.append(df3[c])
    used_columns = np.array(used_columns, dtype=np.float32)
    abs_vel_3 = np.linalg.norm(used_columns, ord=2, axis=0)
    abs_vel_1_processed = p.process_fast(abs_vel_1)
    abs_vel_2_processed = p.process_fast(abs_vel_2)
    abs_vel_3_processed = p.process_fast(abs_vel_3)
    plt.subplot(3, 2, 1)
    plt.plot(list(range(len(abs_vel_1))), abs_vel_1)
    plt.title("Near fall")
    plt.subplot(3, 2, 3)
    plt.plot(list(range(len(abs_vel_2))), abs_vel_2)
    plt.title("Fall")
    plt.subplot(3, 2, 5)
    plt.plot(list(range(len(abs_vel_3))), abs_vel_3)
    plt.title("Normal data (Processed)")
    plt.subplot(3, 2, 2)
    plt.plot(list(range(len(abs_vel_1_processed))), abs_vel_1_processed)
    plt.title("Near fall (Processed)")
    plt.subplot(3, 2, 4)
    plt.plot(list(range(len(abs_vel_2_processed))), abs_vel_2_processed)
    plt.title("Fall (Processed)")
    plt.subplot(3, 2, 6)
    plt.plot(list(range(len(abs_vel_3_processed))), abs_vel_3_processed)
    plt.title("Normal data")
    plt.show()
    original = DFT_slow(abs_vel_1)
    original = np.real(original)
    original[0] = 0
    N = len(original)
    modified = [x if abs(N//2-i) > N*(0.98/2) else 0 for i,x in enumerate(original)]
    modified = np.real(modified)
    modified[0] = 0
    plt.subplot(2, 1, 1)
    plt.plot(list(range(len(original))), original)
    plt.subplot(2, 1, 2)
    plt.plot(list(range(len(modified))), modified)
    plt.show()
