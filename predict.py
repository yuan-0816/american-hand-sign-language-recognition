import tensorflow as tf
import numpy as np
import os
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

categorical = ['0',
               '1',
               '2',
               '3',
               '4',
               '5',
               '6',
               '7',
               '8',
               '9',
               'a',
               'b',
               'c',
               'd',
               'e',
               'f',
               'g',
               'h',
               'i',
               'j',
               'k',
               'l',
               'm',
               'n',
               'o',
               'p',
               'q',
               'r',
               's',
               't',
               'u',
               'v',
               'w',
               'x',
               'y',
               'z']



def get_hand_points(img):
    results = hands.process(img)
    points = []
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
        if len(points) > 21:
            points = points[:21]
        elif len(points) < 21:
            dif = 21 - len(points)
            for i in range(dif):
                points.append([0, 0, 0])

        points = np.array(points)


    points_raw = points
    # 標準化
    if points_raw is not None:
        min_x = np.min(points_raw[:, 0])
        max_x = np.max(points_raw[:, 0])
        min_y = np.min(points_raw[:, 1])
        max_y = np.max(points_raw[:, 1])
        for i in range(len(points_raw)):
            points_raw[i][0] = (points_raw[i][0] - min_x) / (max_x - min_x)
            points_raw[i][1] = (points_raw[i][1] - min_y) / (max_y - min_y)


    return points_raw






if __name__ == '__main__':
    model = tf.keras.models.load_model('model/best.h5py')
    model.summary()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while (True):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        point = get_hand_points(frame)

        if point is not None:
            point = np.expand_dims(point, axis=0)
            # print(point.shape)

            predictions = model.predict(point)
            predicted_labels = np.argmax(predictions, axis=1)
            cv2.putText(frame, str(categorical[predicted_labels[0]]), (10, 130),
                        2, 5, (255, 0, 0), thickness=2)


        cv2.imshow("", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()






    # dataset_root = "./point_datasets"
    # # 获取所有子文件夹的名称
    # subfolders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    #
    # # 从每个子文件夹中随机选择一个.npy文件
    # for subfolder in subfolders:
    #     folder_path = os.path.join(dataset_root, subfolder)
    #     files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    #
    #     if files:
    #         # 随机选择一个文件
    #         random_file = np.random.choice(files)
    #         file_path = os.path.join(folder_path, random_file)
    #
    #         # 使用 np.load 读取数据
    #         data = np.load(file_path)
    #         adjusted_data = np.expand_dims(data, axis=0)
    #
    #         predictions = model.predict(adjusted_data)
    #         predicted_labels = np.argmax(predictions, axis=1)
    #         print(categorical[predicted_labels[0]])


