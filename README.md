Animal face diagnosis
=====================

## Description
Using KAZE features, Bag-of-Visual Words and Histgram intersection, this tool diagnose which animals your face resembles.

## Requirement
- opencv 3.1.0+
- scikit-learn
- numpy 

## Installation
```
git clone https://github.com/Shoriman/Animal-face-diagnosis/tree/master
```

## Usage
- input: Image with face in front
- output: An animal most similar to your face
```
./diagnosis.py input_image
```

## Example
```zsh
%./diagnosis.py face_img_sample.tif
Your face looks like mouse_face!
```

## Files in this repository
- animal_faces_train: Training dataset used to create Bag-of-Visual Words
- calc_BoVW_hist.py: This program calculate Bag-of-Visual Words and Image feature histograms from Animal-face-diagnosis/animal_faces_train
- animal_face_vw: Bag-of-Visual Words calculated by Animal-face-diagnosis/calc_BoVW_hist.py (required for Animal-face-diagnosis/diagnosis.py)
- animal_face_histgrams: Image feature histograms calculated by Animal-face-diagnosis/calc_BoVW_hist.py (required for Animal-face-diagnosis/diagnosis.py)
- functions.py: Module containing several functions such as Bag-of-Visual Words creation
- diagnosis.py: Execute animal-face-diagnosis
- haarcascade_frontalface_alt.xml: Pre-trained Haar-like feature classifier (from opencv 3.2.0)
- face_img_sample.tif: A sample of input image



