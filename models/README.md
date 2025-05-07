# Model Training and Evaluation

This README explains the purpose and usage of the Python scripts in the `models/` directory. These scripts handle the training, testing, and evaluation of the CNN used for binary image classification (person vs. object).

## Prerequisites

* Dataset folders (`data/train/`, `data/val/`, `data/test/`) must already be generated using the scripts in the project root.
* Trained model will be saved to the models/ directory.

## Scripts Overview

1. **`face_detection.py`**: Defines the CNN model architecutre.
2. **`train.py`**: Trains the CNN using the training and validation sets.
3. **`test.py`**: Loads the trained model and runs inference on the test set. Saves predictions for later evaluation.
4. **`evaluate.py`**: Loads the predictions, computes key performance metrics (accuracy, precision, recall, F1), and displays a confusion matrix plot.

## Usage

1. **Train the model**

   ```bash
   python train.py
   ```

   This script will:

   * Load the training and validation data from the `data/` directory.
   * Train the CNN model.
   * Save the best model as `best_model.pth`.
   * Save a plot of training and validation accuracy/loss.

2. **Test the model**

   ```bash
   python test.py
   ```

   This script will:

   * Load the saved model `best_model.pth`.
   * Run inference on the test set.
   * Save predictions and labels to `test_outputs.npz`.

3. **Evaluate the model**

   ```bash
   python evaluate.py
   ```
   
   This script will:
   * Load predictions from `test_outputs.npz`.
   * Compute accuracy, precision, recall, and F1 score.
   * Display and save a confusion matrix to `confusion_matrix.png`
## Directory Structure

```
models/
├── face_detection.py
├── train.py
├── test.py
├── evaluate.py
├── best_model.pth
├── training_history.png
├── test_outputs.npz
├── confusion_matrix.png
└── README.md
```

## Notes

* All scripts assume a consistent folder structure under `data/` with `person/` and `object/` subfolders for each split.
* Make sure `best_model.pth` is available before running `test.py` or `evaluate.py`

---

Happy experimenting!

