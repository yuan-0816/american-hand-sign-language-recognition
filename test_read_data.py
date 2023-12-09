import os
import numpy as np

def load_data(data_folder):
    data = []
    labels = []

    # 創建標籤映射
    label_mapping = {str(i): i for i in range(10)}  # 數字標籤映射
    label_mapping.update({chr(ord('a') + i): i + 10 for i in range(26)})  # 字母標籤映射

    for label in os.listdir(data_folder):
        label_path = os.path.join(data_folder, label)
        if os.path.isdir(label_path):
            # 讀取資料夾中的所有 npy 檔案
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if file_name.endswith(".npy"):
                    # 讀取 npy 檔案
                    npy_data = np.load(file_path)
                    # 加入資料和對應的標籤
                    data.append(npy_data)
                    labels.append(label_mapping[label])

    # 轉換為 NumPy 陣列
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

# 指定你的資料夾路徑
data_folder = 'point_datasets'

# 讀取資料
train_x, train_y = load_data(data_folder)

# 打印形狀
# print(train_x.shape)
for i in range(0, 36):
    x = i + 1260*i
    print(train_y[x])
