# CS549_Person_Object_Detection

# Face Detection Model

This project aims to develop a convolutional neural network (CNN) that can accurately determine whether an image contains a human face or not. The model is trained on a dataset of celebrity faces (CelebA) and non-human objects (from OpenImagesV7 and CIFAR-10).

## Project Structure

```
project-root/
├── .gitignore
├── README.md
├── requirements.txt
├── celeba_raw/               # (ignored by .gitignore)
├── processed/                # (ignored by .gitignore)
├── data/
│   ├── README.md
│   ├── train/               
│   │   ├── person/          # Face images
│   │   └── object/          # Object images
│   ├── val/                 
│   │   ├── person/          # Face images
│   │   └── object/          # Object images
│   └── test/                
│       ├── person/          # Face images
│       └── object/          # Object images
├── scripts/
│   ├── README.md
│   ├── cifar/               
│   │   └── cifardownloadandsplit.py
│   └── kaggle/              
│       ├── kaggledatadownload.py
│       ├── kaggledatasplit.py
│       └── preprocess.py     
├── models/
│   ├── README.md
│   ├── face_detection.py
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   ├── best_model.pth       # (generated when model is trained)
│   ├── training_history.png # (generated when model is trained)
│   ├── test_outputs.npz     # (generated during model testing)
│   └── confusion_matrix.png # (generated during model evaluation)
```