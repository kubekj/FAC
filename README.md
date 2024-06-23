# Facial Attribute Classification (FAC)

This repository contains a comprehensive project on Facial Attribute Classification (FAC), focusing on the analysis, comparison, and evaluation of [Facer](https://github.com/FacePerceiver/facer) and [SwinFace](https://github.com/lxq1000/SwinFace). The projects have been tested against the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Setup Instructions

### Facer Model
To run the Facer model, execute the steps in `experiments.ipynb` to run the model.

### SwinFace Model
To run the SwinFace model, follow these steps:
1. Download SwinFace locally.
2. Replace `inference.py` with the version included in the FAC repository.
3. Complete the setup by following the instructions in the [SwinFace](https://github.com/lxq1000/SwinFace) `.readme` file.
  - Download the model
  - Adjust the paths in `inference.py` file

### CelebA Dataset
Download the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned and cropped dataset and store it in a folder above the FAC repository directory.
