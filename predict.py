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
import numpy as np
import cv2
import mediapipe as mp
import drawing_style as ds


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

categorical = ['0', '1', '2', '3', '4', '5',
               '6', '7', '8', '9', 'a', 'b',
               'c', 'd', 'e', 'f', 'g', 'h',
               'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z']

def get_hand_points(img):

    points = []
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                points.append([x, y, z])

            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                ds.get_default_hand_landmarks_style(),
                ds.get_default_hand_connections_style()
            )
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

    # 標準化
    points_raw = points
    if points_raw is not None:
        min_x = np.min(points_raw[:, 0])
        max_x = np.max(points_raw[:, 0])
        min_y = np.min(points_raw[:, 1])
        max_y = np.max(points_raw[:, 1])
        for i in range(len(points_raw)):
            points_raw[i][0] = (points_raw[i][0] - min_x) / (max_x - min_x)
            points_raw[i][1] = (points_raw[i][1] - min_y) / (max_y - min_y)

    return points_raw, img



if __name__ == '__main__':
    try:
        model = tf.keras.models.load_model('model/ASL_Recognition.h5py')
    except:
        print(f'model does not exist!')
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while (True):
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            point, draw_img = get_hand_points(frame)
            draw_img = cv2.flip(draw_img, 1)

            if point is not None:
                point = np.expand_dims(point, axis=0)
                # print(point.shape)
                predictions = model.predict(point)
                predicted_labels = np.argmax(predictions, axis=1)

                cv2.putText(
                    img=draw_img,
                    text=str(categorical[predicted_labels[0]]),
                    org=(10, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(254, 254, 254),
                    thickness=6)

                cv2.putText(
                    img=draw_img,
                    text=str(categorical[predicted_labels[0]]),
                    org=(10, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(192, 255, 48),
                    thickness=2,
                    lineType=cv2.LINE_AA)

            cv2.imshow("", draw_img)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



    # 讀取資料集預測
    # dataset_root = "./point_datasets"
    #
    # subfolders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    #
    # for subfolder in subfolders:
    #     folder_path = os.path.join(dataset_root, subfolder)
    #     files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    #
    #     if files:
    #
    #         random_file = np.random.choice(files)
    #         file_path = os.path.join(folder_path, random_file)
    #
    #         data = np.load(file_path)
    #         adjusted_data = np.expand_dims(data, axis=0)
    #
    #         predictions = model.predict(adjusted_data)
    #         predicted_labels = np.argmax(predictions, axis=1)
    #         print(categorical[predicted_labels[0]])


