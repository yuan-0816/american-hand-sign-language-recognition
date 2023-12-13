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
import copy


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

categorical = ['0', '1', '2', '3', '4', '5',
               '6', '7', '8', '9', 'a', 'b',
               'c', 'd', 'e', 'f', 'g', 'h',
               'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z']

_CYAN = (192, 255, 48)
_WHITE = (224, 224, 224)
_BLOCK = (0, 0, 0)



def calc_bounding_rect(image_width, image_height, landmarks):

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def standardization_points(landmarks, isLeft):
    points = []
    if isLeft == "Left":
        for _, landmark in enumerate(landmarks.landmark):
            x = landmark.x
            y = landmark.y
            z = landmark.z
            points.append([x, y, z])
    else:
        for _, landmark in enumerate(landmarks.landmark):
            x = landmark.x * (-1)
            y = landmark.y
            z = landmark.z
            points.append([x, y, z])

    if len(points) > 21:
        points = points[:21]
    elif len(points) < 21:
        dif = 21 - len(points)
        for i in range(dif):
            points.append([0, 0, 0])

    points = np.array(points)

    if points is not None:
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        for i in range(len(points)):
            points[i][0] = (points[i][0] - min_x) / (max_x - min_x)
            points[i][1] = (points[i][1] - min_y) / (max_y - min_y)

    return points






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

        ret, frame = cap.read()
        image_width, image_height = frame.shape[1], frame.shape[0]

        while (True):
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.flip(frame, 1)

            draw_img = copy.deepcopy(frame)

            results = hands.process(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    point = standardization_points(hand_landmarks, handedness.classification[0].label[0:])

                    point = np.expand_dims(point, axis=0)
                    # print(point.shape)
                    predictions = model.predict(point)
                    predicted_labels = np.argmax(predictions, axis=1)

                    mp_drawing.draw_landmarks(
                        draw_img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        ds.get_default_hand_landmarks_style(),
                        ds.get_default_hand_connections_style()
                    )

                    bounding_rect = calc_bounding_rect(image_width, image_height, hand_landmarks)
                    if bounding_rect:
                        # 外接矩形
                        cv2.rectangle(
                            draw_img,
                            (bounding_rect[0], bounding_rect[1]),
                            (bounding_rect[2], bounding_rect[3]),
                            _CYAN, 1)

                        cv2.rectangle(draw_img, (bounding_rect[0], bounding_rect[1]), (bounding_rect[2], bounding_rect[1] - 22),
                                      _CYAN, -1)

                        info_text = handedness.classification[0].label[0:] + ": " + str(categorical[predicted_labels[0]])

                        cv2.putText(draw_img, info_text, (bounding_rect[0] + 5, bounding_rect[1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BLOCK, 1, cv2.LINE_AA)


            cv2.imshow("", draw_img)

            if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


