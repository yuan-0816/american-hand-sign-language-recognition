import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from point_net import PointNet



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



if __name__=="__main__":

    if not os.path.exists('training_history'):
        os.makedirs('training_history')

    point_data_folder_path = 'point_datasets'
    try:
        if not os.path.exists(point_data_folder_path):
            raise FileNotFoundError(f"The folder '{point_data_folder_path}' does not exist!")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    train_data_path = './data/train_data.npy'
    train_labels_path = './data/train_labels.npy'
    try:
        if not (os.path.isfile(train_data_path) and os.path.isfile(train_labels_path)):
            train_data, train_labels = load_data(point_data_folder_path)
            np.save(file=train_data_path, arr=train_data)
            np.save(file=train_labels_path, arr=train_labels)
    except:
        pass
    else:
        train_data = np.load('./data/train_data.npy')
        train_labels = np.load('./data/train_labels.npy')
        train_labels = to_categorical(train_labels, num_classes=36)

        tf.random.set_seed(42)

        train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

        num_classes = 36
        point_net_model = PointNet(num_classes)

        optimizer = Adam(learning_rate=0.0001)
        point_net_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        csv_logger = CSVLogger(os.path.join('training_history', 'training_history.csv'))


        point_net_model.fit(train_data, train_labels,
                            epochs=20,
                            validation_data=(test_data, test_labels),
                            callbacks=[early_stopping, csv_logger])

        point_net_model.save('model/ASL_Recognition.h5py')

