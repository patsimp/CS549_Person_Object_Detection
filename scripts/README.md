# Project Data Setup

This README explains how to download and partition the image dataset for use with the provided Python scripts.

## Prerequisites

* Python 3.7 or higher
* Required Python packages listed in `requirements.txt` (install via `pip install -r requirements.txt`)
* A valid Kaggle API token configured (see [Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials))

## Scripts Overview

1. **`kaggledatadownload.py`**: Downloads the raw image dataset from Kaggle.
2. **`kaggledatasplit.py`**: Splits the downloaded dataset into training, validation, and test sets.
4. **`sample_objects.py`**: Downloads CIFAR-10 dataset using 'torchvision.datasets' and randomly samples and resizes the object images to match the face images.

## Usage

1. **Download the dataset**

   ```bash
   python kaggledatadownload.py
   ```

   This script will:

   * Connect to the Kaggle API using your credentials.
   * Download the specified dataset into the `data/raw/` directory.

2. **Partition the dataset**

   ```bash
   python kaggledatasplit.py
   ```

   This script will:

   * Read the images from `data/raw/`.
   * Split them into `data/train/`, `data/val/`, and `data/test/` directories according to the predefined ratios.

## Directory Structure

```
project-root/
├── data/
│   ├── raw/                 # Downloaded images
│   ├── train_small/         # Training set images
│   ├── ├── person/          # 2000 face images
│   ├── ├── object/          # 2000 object images
│   ├── val_small/           # Validation set images
│   ├── ├── person/          # 500 face images
│   ├── ├── object/          # 500 object images
│   └── test_small/          # Test set images
│   ├── ├── person/          # 500 face images
│   ├── ├── object/          # 500 object images
├── kaggledatadownload.py
├── kaggledatasplit.py
├── sample_faces.py
├── sample_objects.py
├── requirements.txt
└── README.md
```

## Notes

* Ensure you have sufficient disk space before downloading large datasets.
* You can adjust the split ratios by modifying the constants at the top of `kaggledatasplit.py`.
* For any issues, please open an issue on the project repository or contact the maintainer.

---

Happy experimenting!

