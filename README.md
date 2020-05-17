# Image-Classifier
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/urastogi885/image-classifier/blob/master/LICENSE)

## Overview

This project trains a model to classify an image in a dataset into a dog or a cat. The dataset used for the project can be found [*here*](https://drive.google.com/open?id=1So8fzjoRUJGHuGE_Jw-QsY_21F_g4YMb).

## Dependencies

- Python3
- Python3 Libraries: tensorflow, keras, matplotlib, glob, sys, os, pillow

## Install Dependencies

- Install Python3, Python3-tk, and the necessary libraries: (if not already installed)

```
sudo apt install python3
sudo apt install python3-pip
pip3 install matplotlib, glob
pip3 install pillow==6.2.1
pip3 install tensorflow==1.15.0
pip3 install keras
```

- Check if your system successfully installed all the dependencies
- Open terminal using ```Ctrl+Alt+T``` and enter ```python3```.
- The terminal should now present a new area represented by ```>>>``` to enter python commands
- Now use the following commands to check libraries: (Exit python window using ```Ctrl + Z``` if an error pops up while
running the below commands)

```
import tesnorflow
import keras, glob
import matplotlib
```
## Run

- Download each of the dataset mentioned in the [*Overview Section*](https://github.com/urastogi885/image-classifier#overview).
- It is recommended that you save the dataset within outer-most directory-level of the project otherwise it will become 
too cumbersome for you to reference the correct location of the file.
- Using the terminal, clone this repository and go into the project directory, and run the main program:

```
https://github.com/urastogi885/image-classifier
cd image-classifier/Code
python3 image-classifier.py <dataset_location>
```

- Note that the dataset location is relative to the current working directory.
- The dataset location is provided for the training dataset. See example below
- If you have a compressed version of the project, extract it, go into project directory, open the terminal by 
right-clicking on an empty space, and type:

```
cd Code/
python3 image-classifier.py <dataset_location>
```
- For instance:
```
python3 image-classifier.py ../dogs-vs-cats/train/
```
