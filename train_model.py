import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
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


    # data_folder = 'point_datasets'
    # train_data, train_labels = load_data(data_folder)
    # np.save(file='./data/train_data.npy', arr=train_data)
    # np.save(file='./data/train_labels.npy', arr=train_labels)



    train_data = np.load('./data/train_data.npy')
    train_labels = np.load('./data/train_labels.npy')
    train_labels = to_categorical(train_labels, num_classes=36)

    # train_labels->one hot
    # idx = np.random.permutation(len(data))
    # x, y = data[idx], classes[idx]

    # from sklearn.utils import shuffle
    # X, y = shuffle(X, y, random_state=0)

    # 設定隨機種子以確保實驗結果的可重複性
    tf.random.set_seed(42)

    # 將資料切分為訓練集和驗證集
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    # 初始化 PointNet 模型
    num_classes = 36  # 你的分類數量
    point_net_model = PointNet(num_classes)

    # 編譯模型
    optimizer = Adam(learning_rate=0.0001)
    point_net_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 訓練模型
    point_net_model.fit(train_data, train_labels,
                        epochs=10,
                        validation_data=(test_data, test_labels),)

    predictions = point_net_model.predict(train_data)
    predicted_labels = np.argmax(predictions, axis=1)
    train_labels = np.argmax(train_labels, axis=1)
    accuracy = accuracy_score(train_labels, predicted_labels)
    print(f'Accuracy: {accuracy}')

    confusion_mat = confusion_matrix(train_labels, predicted_labels)
    print('Confusion Matrix:')
    print(confusion_mat)

    point_net_model.save('model/best.h5py')


