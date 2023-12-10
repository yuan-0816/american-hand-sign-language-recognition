import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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


def show_training_history(path):

    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The training history '{path}' does not exist!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    else:

        history_df = pd.read_csv(path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(history_df['loss'], label='Training Loss')
        ax1.plot(history_df['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(history_df['accuracy'], label='Training Accuracy')
        ax2.plot(history_df['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()

        plt.show()



if __name__ == '__main__':

    show_training_history('training_history/training_history.csv')


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
        point_net_model = tf.keras.models.load_model('model/best.h5py')
        train_data = np.load('./data/train_data.npy')
        train_labels = np.load('./data/train_labels.npy')
        test_labels_encoded = np.argmax(train_labels, axis=1)  # 將 one-hot 編碼轉換為未編碼

        # 進行預測
        predictions = point_net_model.predict(test_data)
        predicted_labels = np.argmax(predictions, axis=1)

        # 計算準確率
        accuracy = accuracy_score(test_labels_encoded, predicted_labels)
        print(f'Accuracy: {accuracy}')

        # 繪製混淆矩陣
        confusion_mat = confusion_matrix(test_labels_encoded, predicted_labels)
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = [str(i) for i in range(36)]  # 假設有 36 類
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()