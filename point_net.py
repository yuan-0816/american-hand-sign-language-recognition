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
import tensorflow as tf
from tensorflow.keras import layers, Model

class PointNet(Model):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()

        # Shared MLP
        self.mlp = tf.keras.Sequential([
            layers.Conv1D(64, 1, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 1, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(1024, 1, activation='relu'),
            layers.BatchNormalization()
        ])

        # Global max pooling
        self.global_max_pool = layers.GlobalMaxPooling1D()

        # Fully connected layers
        self.fc_layers = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, x):
        # Permute to (batch_size, num_points, num_channels)
        x = tf.transpose(x, perm=[0, 2, 1])
        # Apply shared MLP
        x = self.mlp(x)
        # Global max pooling
        x = self.global_max_pool(x)
        # Fully connected layers
        x = self.fc_layers(x)
        return x

    def get_model():
        return PointNet(name='point_net')

if __name__ == '__main__':
    # Creating an instance of PointNet
    num_classes = 36  # Number of classes in the classification task
    model = PointNet(num_classes)

    # Generating a random input point cloud tensor
    batch_size = 20
    num_points = 1024
    num_channels = 3
    input_points = tf.random.normal((batch_size, num_points, num_channels))

    # Forward pass
    output = model(input_points)

    model.summary()

    print("Output shape:", output.shape)
