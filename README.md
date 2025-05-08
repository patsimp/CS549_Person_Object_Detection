# CS549_Person_Object_Detection

# Face Detection Model

This project aims to develop a convolutional neural network (CNN) that can accurately determine whether an image contains a human face or not. The model is trained on a dataset of celebrity faces (CelebA) and non-human objects (from OpenImagesV7 and CIFAR-10).

## Project Structure

```
project-root/
├── data/
│   ├── raw/                 # Downloaded images
│   ├── train/               # Training set images
│   ├── ├── person/          # Face images
│   ├── ├── object/          # Object images
│   ├── val/                 # Validation set images
│   ├── ├── person/          # Face images
│   ├── ├── object/          # Object images
│   └── test/                # Test set images
│   ├── ├── person/          # Face images
│   ├── ├── object/          # Object images
├── scripts/
│   ├── cifar/               # CIFAR dataset scripts
│   │   └── cifardownloadandsplit.py
│   └── kaggle/              # Kaggle dataset scripts
│       ├── kaggledatadownload.py
│       └── kaggledatasplit.py
│       └── preprocess.py     
├── requirements.txt
└── README.md
```