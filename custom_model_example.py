import cv2
import sys
from ultralytics import YOLO

# Load your custom trained model
model = YOLO('path/to/your/custom/model.pt')  # Replace with your trained model path

# Your existing video processing code...
# Make sure to update the class indices if you added e-rickshaw as a new class