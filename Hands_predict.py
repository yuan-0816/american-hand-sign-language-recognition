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



def calc_bounding_rect(image_width, image_height, landmarks):

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     _CYAN, 3)
    return image

def get_hand_points(img):

    points = []
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if (handedness.classification[0].label[0:]) == "Left":
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    points.append([x, y, z])
            else:
                for i in range(21):
                    x = (-1)*hand_landmarks.landmark[i].x
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


def test_hand(img):
    points = []
    results = hands.process(img)

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            if (handedness.classification[0].label[0:]) == "Left":
                print('Left')

        # a =  getattr(results, 'multi_hand_landmarks')
        # b = getattr(results, 'multi_handedness')



        # for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            #

            # for i in range(21):
            #     x = hand_landmarks.landmark[i].x
            #     y = hand_landmarks.landmark[i].y
            #     z = hand_landmarks.landmark[i].z
            #     points.append([x, y, z])
            #
            # brect = calc_bounding_rect(image_width, image_height, hand_landmarks)
            # img = draw_bounding_rect(True, img, brect)
            #
            # mp_drawing.draw_landmarks(
            #     img,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     ds.get_default_hand_landmarks_style(),
            #     ds.get_default_hand_connections_style()
            # )





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

            point, draw_img = get_hand_points(frame)

            # draw_img = cv2.flip(draw_img, 1)

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


