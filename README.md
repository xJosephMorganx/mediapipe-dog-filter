# Dog Filter with MediaPipe and OpenCV

This project applies a dog-style face filter using a webcam. It uses **MediaPipe Face Landmarker** to detect facial landmarks and **OpenCV** to display the camera feed in real time while placing transparent PNG images over the face.

## How It Works

The program opens the webcam, detects a face, and gets facial landmarks using MediaPipe. Based on those points, it calculates the approximate position of the forehead, nose, and mouth to place the filter elements:

- Dog ears above the head
- Dog nose over the real nose
- Dog tongue when the mouth is detected as open

The images are placed over the video using alpha transparency, so the filter files must be transparent PNG images.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```txt
mediapipe
numpy
opencv-python
```

## Usage

Run the program with:

```bash
python filtro_perro.py
```

Press `ESC` to close the window.

## Required Files

To work correctly, the project needs the following files in the same folder as the Python script:

```text
face_landmarker.task
dog_ears.png
dog_nose.png
dog_tongue.png
```

## Asset Notice

The image assets used for the filter (`dog_ears.png`, `dog_nose.png`, and `dog_tongue.png`) are **not included** in this repository because their license could not be verified.

To run the project, add your own transparent PNG files with the same names, or use images with a clear license that allows use and redistribution.

## Notes

This project was created for educational purposes to practice computer vision, facial landmark detection, and real-time image overlay using Python.