import numpy as np
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_datasets(root):
    os.chdir(root)
    directories = os.listdir()
    paths = []
    for directory in directories:
        pointdata = [f for f in os.listdir(directory) if f.lower().endswith('.npy')]
        if len(pointdata) >= 2:
            chosen_data = random.sample(pointdata, 1)
            for data in chosen_data:
                data_path = os.path.join(root, directory, data) # root + "/" + directory + "/" + data
                paths.append(data_path)
                # print(paths)
    return paths


def test():

    # 数据集根目录
    dataset_root = "point_datasets"

    # 获取所有子文件夹的名称
    subfolders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]

    # 从每个子文件夹中随机选择一个.npy文件
    for subfolder in subfolders:
        folder_path = os.path.join(dataset_root, subfolder)
        files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

        if files:
            # 随机选择一个文件
            random_file = np.random.choice(files)
            file_path = os.path.join(folder_path, random_file)

            # 使用 np.load 读取数据
            data = np.load(file_path)

            # 这里可以使用读取到的数据进行处理
            # print(f"Loaded data from {file_path}")
            # print(data.shape)
            print(type(data))

def show_datasets():

    # 读取的点云数据
    data = np.array([
        [6.45287787e-01, 1.00000000e+00, -4.18567453e-07],
        [8.17763770e-01, 8.31491857e-01, -7.27760941e-02],
        [8.75000000e-01, 6.03611638e-01, -9.55623761e-02],
        [6.48971204e-01, 4.40025110e-01, -1.13319330e-01],
        [3.94433316e-01, 3.87702789e-01, -1.24489680e-01],
        [6.73070406e-01, 4.21780500e-01, -2.25688089e-02],
        [5.45846118e-01, 2.12003925e-01, -7.16625303e-02],
        [4.46319847e-01, 9.96404892e-02, -9.55499411e-02],
        [3.37023579e-01, 0.00000000e+00, -1.13675617e-01],
        [4.83330487e-01, 4.96175674e-01, -2.17967853e-02],
        [3.66950439e-01, 3.69718586e-01, -1.01649903e-01],
        [5.34258194e-01, 5.35111580e-01, -1.08138725e-01],
        [6.08383264e-01, 5.98212184e-01, -9.05520394e-02],
        [3.40716764e-01, 5.80289463e-01, -3.24454345e-02],
        [2.28912263e-01, 4.89282229e-01, -1.38828188e-01],
        [4.28582487e-01, 6.54614531e-01, -1.09169669e-01],
        [4.88807700e-01, 7.12072378e-01, -6.09947629e-02],
        [2.04671766e-01, 6.84229673e-01, -4.85786013e-02],
        [1.25000000e-01, 5.92884739e-01, -1.23207897e-01],
        [2.83951107e-01, 7.05565968e-01, -9.36328024e-02],
        [3.34080823e-01, 7.67129246e-01, -5.45637198e-02]
    ])

    print(type(data))

    # 创建 3D 子图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    # dir = 'point_datasets'
    # random_point_datas = check_datasets(dir)
    # show_datasets(random_point_datas)
    # data = np.load('point_datasets/0/hand1_0_bot_seg_1_cropped_Rot_1.npy')
    # print(data.shape)

    # print(data)


    test()
    # show_datasets()


    # arrays = {}
    # for filename in os.listdir(dir):
    #     if filename.endswith('.npy'):
    #         arrays[filename] = load_array(filename)




    # print(random_point_datas)

    # data = np.load('point_datasets\\5\\hand2_5_bot_seg_4_cropped_Rot_5_Axis_z.npy')
    # print(data)

    # file_paths = [
    #     'point_datasets/0/hand1_0_bot_seg_3_cropped_Scale_0_Axis_y.npy',
    #     'point_datasets/1/hand4_1_bot_seg_5_cropped_Rot_14_Axis_y.npy',
    #     'point_datasets/2/hand3_2_dif_seg_4_cropped_Rot_-9_Axis_y.npy'
    # ]
    #
    #
    # print(type(random_point_datas))
    # print(type(file_paths))

    # for random_point_data in random_point_datas:
    #     data = np.load(random_point_datas)
    #     print(data)





