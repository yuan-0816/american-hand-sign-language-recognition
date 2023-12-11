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

## Datasets
This is the American Sign Language dataset from Kaggle, consisting of a total of 36 categories, including numbers 0-9 and English letters A-Z.

https://www.kaggle.com/datasets/ayuraj/asl-dataset/data
![](https://github.com/yuan-0816/american-hand-sign-language-recognition/blob/main/doc/result/all_image.png)

put the image datasets in [datasets](https://github.com/yuan-0816/american-hand-sign-language-recognition/tree/main/datasets/data) folder
Similar to the file structure mentioned earlier, or you may need to adjust the code for file path references.





