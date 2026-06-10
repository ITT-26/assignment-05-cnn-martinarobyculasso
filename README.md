[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cMaQVOgt)

# Assignment 5: Hand Pose Detection with a CNN

Assignment 5 for the Interactive Techniques and Technologies course (ITT), Universität Regensburg.

Author: Martina Roby Culasso

---

Each folder contains an `info.txt` file with a description of the files and relevant notes.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Exercise 1 - Hyperparameter Exploration: Image Color Mode

Explores the effect of image color mode on CNN prediction accuracy and convergence speed. Five color spaces are tested: RGB, Grayscale, HSV, LAB, and YCrCb. A separate model is trained for each using a subset of the HaGRID dataset, and results are compared via validation accuracy, validation loss, epochs until convergence, and confusion matrices.

The CNN architecture is based on the one provided in class, with early stopping and learning rate reduction callbacks.

**Usage:**

(If you want to re-train the models yourself)

Open `01-hyperparameters/image_color_mode.ipynb` and run all cells. Before running, download the HaGRID dataset or a subset (for example, the subset from GRIPS) and update `DATASET_PATH` at the top of the notebook to point to your local copy.

---

## Exercise 2 - Gathering a Dataset

Extends the HaGRID dataset with custom images of five hand gestures: `like`, `dislike`, `stop`, `rock`, and `peace`. Images were captured and annotated using a modified version of the annotater tool from [Patrick Graf's repository](https://github.com/ITT-26/assignment-05-cnn-PatrickGraf99/blob/main/02-dataset/annotater.py), with a countdown timer added to allow stepping back from the camera before the frame is frozen, making it easier to capture images at a similar distance to the HaGRID dataset.

Annotations are stored in `annot-martinarobyculasso.json`, compatible with the HaGRID annotation format. The images were manually organized into per-category folders to match the HaGRID directory structure, as well as the annotations file, allowing the dataset loading code from Exercise 1 to be reused directly.

A CNN trained on Grayscale (best performing color mode from Exercise 1, same architecture as Exercise 1) is used to make predictions on the custom images. Results are visualized as a confusion matrix saved as `conf-matrix.png`.

**Usage:**

(If you want to repeat the results)

Open `02-dataset/evaluation.ipynb` and run all cells. Before running, download the HaGRID dataset or a subset (for example, the subset from GRIPS) and update `DATASET_PATH` at the top of the notebook to point to your local copy.

---

## Exercise 3 - Gesture-controlled Camera App

A gesture-controlled camera application that uses a CNN to recognize hand gestures in real time via webcam. Due to other responsibilities and being away travelling, I had not enough time to get the gesture recognition working reliably in practice (I'm sorry).

The model was trained on a subset of the HaGRID dataset using Grayscale images (best performing color mode from Exercise 1), with the same base architecture as in class. While the model performed well on the HaGRID test/validation set, it did not generalize well to my webcam input. Using a physical ArUco board as a bounding box for the region of interest was also attempted, but did not improve results significantly. I suppose I'm doing something wrong when processing the captured images, but I couldn't figure it out on time.

As a result, the submitted version maps keyboard keys to gestures so that the camera effects can at least be tested:

| Key | Gesture | Action |
|-----|---------|--------|
| `1` | like | Starts a selfie countdown |
| `2` | stop | Toggles portrait mode (center stays sharp, background blurred) |
| `3` | rock | Toggles sepia filter |
| `Space` | — | Attempts gesture prediction on the current frame (used for debugging, in the actual app it would be automatic or trigerred with the ArUco board) |

**Usage:**

```bash
cd 03-camera-app
python camera_app.py
python camera_app.py --path ./pictures --timer 3
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--path` | Path to save captured photos | `.` |
| `--timer` | Countdown duration in seconds | `5` |

**Controls:**

| Key | Action |
|-----|--------|
| `1` / `2` / `3` | Trigger gesture |
| `Space` | Predict gesture from current frame |
| `Q` | Quit |

> Requires a webcam.