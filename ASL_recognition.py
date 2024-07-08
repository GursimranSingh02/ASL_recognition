import streamlit as st
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize HandDetector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model_ASL/keras_model.h5", "Model_ASL/labels.txt")

# Constants for image processing
offset = 20
imgSize = 300

# Labels for classification
labels = ["A", "B", "C", "Calm down", "D", "E", "F", "G", "H", "Hello", "I", "I hate you", "I love you", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "Stop", "T", "U", "V", "W", "X", "Y"]

# Initialize video capture
cap = None

# Function to start video capture
def start_camera():
    global cap
    cap = cv.VideoCapture(0)

# Function to stop video capture
def stop_camera():
    global cap
    if cap:
        cap.release()
        cap = None

# Main function for Streamlit app
def main():
    st.title('ASL Gesture Recognition App')  # Title of the app

    # Camera Control
    camera_control = st.toggle('Camera Control')

    # start the camera
    if camera_control:
        start_camera()

    # stop the camera
    else:
        stop_camera()
    

    # Check if camera is running
    if cap is not None and cap.isOpened():
        stframe = st.empty()  # Placeholder for video frames
        while cap.isOpened():
            success, img = cap.read()  # Read a frame from the camera
            if not success:
                st.error('Failed to capture video.')
                break

            imgOutput = img.copy()  # Copy the frame for output
            hands, img = detector.findHands(img)  # Detect hands in the frame
            if hands:
                hand = hands[0]  # Get the first detected hand
                x, y, w, h = hand['bbox']  # Get the bounding box of the hand

                # Create a white image for processing
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the hand region

                # Calculate aspect ratio
                aspectRatio = h / w 

                 # If height is greater than width
                if aspectRatio > 1: 
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize

                # If width is greater than height
                else:  
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Get prediction from the classifier
                prediction, index = classifier.getPrediction(imgWhite)

                # Draw a rectangle and label for the gesture
                cv.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 225, y - offset), (255, 4, 255), cv.FILLED)
                cv.putText(imgOutput, labels[index], (x + 20, y - 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (252, 4, 252), 4)

            # Display the output frame
            stframe.image(imgOutput, channels='BGR', use_column_width=True)
    else:
        st.warning('Toggle the switch to initialize the camera.')

# Entry point for the script
if __name__ == '__main__':
    main()

