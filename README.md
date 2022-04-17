## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is a program for detecting people and counting them. 
Detection is based on photos and videos. 
For faster reading of the photo, the program supports the GPU 
from NVIDIA after the appropriate configuration of the graphics card. 
Without GPU configuration, the program will use the CPU.

## Technologies
* opencv-python 4.5.5.64
* imutils 0.5.4
* yolov4-tiny

## Setup
To run this project, install it locally:
```
$ pip install opencv-python
$ pip install imutils
```
and start program:
```
python run.py
```
