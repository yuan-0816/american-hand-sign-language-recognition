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
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import mediapipe as mp
import drawing_style as ds



class show_data():
    def __init__(self, path="datasets/asl_dataset"):
        self.path = path
        self.sample_path, self.subfolders = self.gets_datasets_path()
        self.img_arr = np.random.choice(self.sample_path, size=3, replace=False)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def gets_datasets_path(self):
        paths = []
        subfolders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        for subfolder in subfolders:
            folder_path = os.path.join(self.path, subfolder)
            files = [f for f in os.listdir(folder_path) if f.endswith(".jpeg")]
            if files:
                random_file = np.random.choice(files)
                file_path = os.path.join(folder_path, random_file)
                paths.append(file_path)
        return paths, subfolders

    def show_all_image_and_label(self, figsize=(10, 8), save_path=None):
        plt.figure(figsize=figsize)
        for i in range(len(self.sample_path)):
            img = plt.imread(self.sample_path[i])
            plt.subplot(6, 6, i + 1)
            plt.imshow(img)
            plt.title(f"Image of {self.subfolders[i]}", fontsize='small')
            plt.axis("off")

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def show_processed_border_image(self, figsize=(10, 8), save_path=None):
        plt.figure(figsize=figsize)
        border_size = 100

        for i in range(len(self.img_arr)):
            img = plt.imread(self.img_arr[i])
            plt.subplot(4, 3, i + 1)
            plt.imshow(img)

            draw_img = img.copy()
            draw_img, _ = self.land_mark(draw_img)

            plt.subplot(4, 3, i + 4)
            plt.imshow(draw_img)

            border_img = cv2.copyMakeBorder(
                img,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            plt.subplot(4, 3, i + 7)
            plt.imshow(border_img)


            draw_border_img= border_img.copy()
            draw_border_img, _ = self.land_mark(draw_border_img)

            plt.subplot(4, 3, i + 10)
            plt.imshow(draw_border_img)

        # plt.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


    def show_3D_point(self, figsize=(10, 5), save_path=None, sign=2):
        # for img_path in self.sample_path:
        img = plt.imread(self.sample_path[sign])
        width, height, _ = img.shape
        img, point = self.land_mark(img)
        point = np.array(point)
        if point.shape[0] > 0:
            fig = plt.figure(figsize=figsize)

            ax_img = fig.add_subplot(121)
            ax_img.imshow(img)
            ax_img.set_title('Original Image')
            ax_img.axis('off')

            ax_3d = fig.add_subplot(122, projection='3d')
            ax_3d.scatter(point[:, 0], point[:, 1]*(-1), point[:, 2]*(-1), c='b', marker='o')
            ax_3d.set_xlabel('X Label')
            ax_3d.set_ylabel('Y Label')
            ax_3d.set_zlabel('Z Label')
            ax_3d.set_title('3D Point Cloud')

            ax_3d.set_box_aspect([width, height, 100])
            ax_3d.view_init(elev=90, azim=270)

            fig.subplots_adjust(wspace=0.5)

            if save_path:
                plt.savefig(save_path)
            plt.show()




    def land_mark(self, img=None):
        points = []
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        results = self.hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    ds.get_default_hand_landmarks_style(),
                    ds.get_default_hand_connections_style())
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    points.append([x, y, z])

        return img, points



if __name__ == '__main__':
    result = show_data("datasets/asl_dataset")
    result.show_all_image_and_label(save_path='doc/result/all_image.png')
    # result.show_processed_border_image(save_path='result/border_image.png')
    # result.show_3D_point(save_path='result/3D_point.png')
