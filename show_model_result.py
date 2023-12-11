"""
　　    　　 ＿＿＿
　　　　　／＞　　  フ
　　　　　|  　_　 _|
　 　　　／` ミ＿xノ
　　 　 /　　　 　 |
　　　 /　 ヽ　　 ﾉ
　 　 │　　|　|　|
　／￣|　　 |　|　|
　| (￣ヽ＿_ヽ_)__)
　＼二つ
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ShowResult:
    def __init__(self, model_path='model/best.h5py'):
        self.model_path = model_path
        self.point_net_model = tf.keras.models.load_model(self.model_path)
        # self.point_net_model.summary()

    def model_sumery(self):
        self.point_net_model.summary()

    def show_training_history(self, path, save_path=None):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The training history '{path}' does not exist!")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        else:
            if not os.path.exists('doc/result'):
                os.makedirs('doc/result')
            history_df = pd.read_csv(path)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.plot(history_df['loss'], label='Training Loss')
            ax1.plot(history_df['val_loss'], label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            ax1.legend()

            ax2.plot(history_df['accuracy'], label='Training Accuracy')
            ax2.plot(history_df['val_accuracy'], label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            plt.show()

    def show_confusion_matrix(self,  figsize=(10, 8), save_path=None):
        if not os.path.exists('doc/result'):
            os.makedirs('doc/result')
        train_data = np.load('datasets/data/train_data.npy')
        train_labels = np.load('datasets/data/train_labels.npy')

        tf.random.set_seed(42)

        train_data, test_data, train_labels, test_labels = train_test_split(
            train_data,
            train_labels,
            test_size=0.2,
            random_state=42)

        predictions = self.point_net_model.predict(test_data)
        predicted_labels = np.argmax(predictions, axis=1)

        # 計算準確率
        accuracy = accuracy_score(test_labels, predicted_labels)
        print(f'Accuracy: {accuracy}')

        # 繪製混淆矩陣
        confusion_mat = confusion_matrix(test_labels, predicted_labels)

        # Set the figure size
        plt.figure(figsize=figsize)

        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix and Accuracy: {accuracy:.4f}')
        plt.colorbar()

        classes = [str(i) for i in range(36)]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__ == '__main__':
    model_path = 'model/ASL_Recognition.h5py'
    result = ShowResult(model_path)
    result.model_sumery()
    result.show_training_history('training_history/training_history.csv',
                                 save_path='doc/result/training_history.png')
    result.show_confusion_matrix(figsize=(10, 8),
                                 save_path='doc/result/confusion_matrix.png')