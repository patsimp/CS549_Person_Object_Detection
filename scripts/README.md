# Project Data Setup

This README explains how to download and partition the image dataset for use with the provided Python scripts.

## Prerequisites

* Python 3.7 or higher
* Required Python packages listed in `requirements.txt` (install via `pip install -r requirements.txt`)
* A valid Kaggle API token configured (see [Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials))

## Scripts Overview

1. **`kaggledatadownload.py`**: Downloads the raw image dataset from Kaggle.
2. **`preprocess.py`**: Processes the data by randomly sampling and resizing the person image
3. **`kaggledatasplit.py`**: Splits the downloaded dataset into training, validation, and test sets.
4. **`cifardownloadandsplit.py`**: Downloads CIFAR-10 dataset using 'torchvision.datasets' and randomly samples and resizes the object images to match the face images.

## Usage

1. **Download the dataset**

   ```bash
   python kaggledatadownload.py
   ```

   This script will:

   * Connect to the Kaggle API using your credentials.
   * Download the specified dataset into the `data/raw/` directory.

2. **Preprocess the kaggle data**

   ```bash
   python preprocess.py --max <max_num>
   ```
   
   This script will:
   * Preprocess the kaggle dataset, resize the images
   * The "max" argument will limit the images process by randomly selecting n images to process, otherwise will process 
   the entire dataset

3. **Partition the dataset**

   ```bash
   python kaggledatasplit.py
   ```

   This script will:

   * Read the images from `data/raw/`.
   * Split them into `data/train/`, `data/val/`, and `data/test/` directories according to the predefined ratios.

4. **Download and Partition the CIFAR-10 dataset**

   ```bash
   python cifardownloadandsplit.py
   ```
   
   This script will:
   * Download the CIFAR-10 dataset using torchvision.datasets.
   * Resise the object images to match the face images.
   * Randomly sample object images and split them into `data/train/object/`, `data/val/object/`, and `data/test/object/` to match the face dataset structure.
## Directory Structure

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

## Notes

* Ensure you have sufficient disk space before downloading large datasets.
* You can adjust the split ratios by modifying the constants at the top of `kaggledatasplit.py`.
* For any issues, please open an issue on the project repository or contact the maintainer.
* Kaggle scripts must be executed in the following order: kaggledatadownload.py, preprocess.py, kaggledatasplit.py

---


