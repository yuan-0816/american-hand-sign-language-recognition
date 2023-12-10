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
import random


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


class show_image_data():
    def __init__(self, path="asl_dataset"):
        self.path = path
        self.sample_path, self.subfolders = self.gets_datasets_path()

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

# TODO　圖片資料集顯示
    def processed_border_image(self, type=0):
        img = plt.imread(self.sample_path[type])
        border_size = 100
        img = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        plt.imshow(img)
        plt.show()





def get_hand_points(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img.flags.writeable = False

    results = hands.process(img)

    # Draw the hand annotations on the image.
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    annotated_image = img.copy()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                points.append([x, y, z])
                # cv2.circle(img_tmp )

    else:
        border_size = 100
        img = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        annotated_image = img.copy()
        results = hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    points.append([x, y, z])
        else:
            points = None

    if points is not None:
        points = np.array(points)
        print(points)
    else:
        print("No point")

def draw_point(root):
    try:
        img = cv2.imread(root)
    except:
        img = root
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = img.shape
    results = hands.process(img)
    tmp_img = img.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                tmp_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return tmp_img





def main():
    path = "asl_dataset"
    img_paths, _ = gets_datasets_path(path)
    type = 16
    # get_hand_points(img_paths[5])

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(img_paths[type]), cv2.COLOR_RGB2BGR))
    plt.title(f"origin image", fontsize='small')
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(processed_border_image(img_paths[type]), cv2.COLOR_RGB2BGR))
    plt.title(f"border image", fontsize='small')
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(draw_point(img_paths[type]))
    plt.title(f"origin image point", fontsize='small')
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(draw_point(processed_border_image(img_paths[type])))
    plt.title(f"origin image point", fontsize='small')
    plt.axis("off")
    plt.show()





if __name__ == '__main__':
    result = show_image_data()
    result.show_all_image_and_label()
    result.processed_border_image()
    # main()
