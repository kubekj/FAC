# Facial Attribute Classification (FAC)

This repository contains a comprehensive project on Facial Attribute Classification (FAC), focusing on the analysis, comparison, and evaluation of [Facer](https://github.com/FacePerceiver/facer) and [SwinFace](https://github.com/lxq1000/SwinFace). The projects have been tested against the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Setup Instructions

<mark> Only do if you want to replicate the whole process of gathering data, if you want to check the comparison stage - skip to the Comparison section below </mark>

### Facer Model

To run the [Facer](https://github.com/FacePerceiver/facer) model, execute the steps in `experiments.ipynb` to run the model and collect the results.

### SwinFace Model

To run the [SwinFace](https://github.com/lxq1000/SwinFace) model, follow these steps:

1. Download SwinFace locally.
2. Replace `inference.py` with the version included in the FAC repository under the `scripts` folder.
3. Complete the setup by following the instructions in the SwinFace `.readme` file:

- Download the model
- Adjust the paths in `inference.py` file

4. Run the `inference.py` file and collect the results.

### CelebA Dataset

Download the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned and cropped dataset and store it in a folder above the FAC repository directory.

## Comparison instructions

### CSV files

To download needed `.csv` for performing the comparisons please go to [Google Drive](https://drive.google.com/drive/folders/1WhWUA6Cb_YZhUrp3EGimH3_XhOhUWn_o?usp=drive_link) that contains them.

### Comparison script

Everything concerning how to run the comparison code is included and described in detail in the experiments.ipynb file.
