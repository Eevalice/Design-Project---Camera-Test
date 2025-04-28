import cv2

# Test webcam indices from 0 to 4
for index in range(5):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Webcam detected at index {index}")
        cap.release()
    else:
        print(f"No webcam detected at index {index}")

import numpy
import scipy
print(numpy.__version__)
print(scipy.__version__)


# creating new environment
#python -m venv new_env
#source new_env/bin/activate  # On Windows: new_env\Scripts\activate
#pip install numpy