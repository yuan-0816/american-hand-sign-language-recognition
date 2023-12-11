# American-Hand-Sign-language-Recognition
This project is based on Edward Roe's work on [ASL recognition using PointNet and Mediapipe](https://medium.com/@er_95882/asl-recognition-using-pointnet-and-mediapipe-f2efda78d089)  
Many thanks to him for his valuable contribution.

---

<p align="center">
  <img src="https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/doc/demo.gif" />
</p>

## File structure
```commandline
hand_sign_language_recognition
├─datasets              
│  ├──asl_dataset           (Sign language image dataset from Kaggle)
│  ├──point_datasets        (The point cloud training dataset generated using Mediapipe)
│  └──data                  (All of the point cloud datasets)
│
├─doc                       (The images in the README)
│ └─result                  (The prediction results)
│
├─model                             
│ └─ASL_Recognition.h5py    (The trained model)
│
├─requirements.txt          (Environment)
├─training_history
│ └─training_history.csv    (The data from the model training process)
│
├─create_point_datasets.py  (Generate point cloud datasets)
├─drawing_style.py          (The drawing style of Mediapipe)
├─point_net.py              (PointNet model)
├─predict.py                (Prediction using a camera)
├─show_datasets.py          (Display the dataset)
├─show_model_result.py      (Display the model result)
└─train_model.py            (Train model)
```


## Environment
#### python 3.9.13

run```pip install -r requirements.txt``` 

## [Datasets](https://www.kaggle.com/datasets/ayuraj/asl-dataset/data)
This is the American Sign Language dataset from Kaggle, consisting of a total of 36 categories, including numbers 0-9 and English letters A-Z.

![](https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/doc/result/all_image.png)

put the image datasets in [datasets](https://github.com/yuan-0816/american-hand-sign-language-recognition/tree/main/datasets/data) folder
Similar to the file structure mentioned earlier, or you may need to adjust the code for file path references.

## Generate point cloud datasets
```commandline
python3 create_point_datasets.py
```
the point cloud datasets will be stored in the [datasets](https://github.com/yuan-0816/american-hand-sign-language-recognition/tree/main/datasets/data) folder

This involves using Mediapipe to detect hand keypoint coordinates, 
totaling 21 points, with each point containing [x, y, z] coordinates.  

You can find more details on [Mediapipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)  

<p align="center">
  <img src="https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/doc/result/3D_point.gif" />
</p>
  
## Model 
we will use part of the PointNet model, because we will not care about things like unordered points or transformation invariance.  

<p align="center">
  <img src="https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/doc/easy_pointnet.png" />
</p>

You can see the code at [point_net.py](https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/point_net.py) 


## Train model

```commandline
python3 train_model.py     
```

The trained model will be stored in the [model](https://github.com/yuan-0816/american-hand-sign-language-recognition/tree/main/model/ASL_Recognition.h5py) folder.

The training history  

<p align="center">
  <img src="https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/doc/result/training_history.png" />
</p>

The confusion matrix  

<p align="center">
  <img src="https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/doc/result/confusion_matrix.png" />
</p>

## Predict
```commandline
python3 predict.py  
```

By yuan :p



