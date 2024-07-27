import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize HandDetector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

# Constants for image processing
offset = 20
imgSize = 300

# Labels for classification
labels = ["A", "B", "C", "Calm down", "D", "E", "F", "G", "H", "Hello", "I", "I hate you", "I love you", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "Stop", "T", "U", "V", "W", "X", "Y"]

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("keras_model.h5", "labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = labels

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                hGap = math.ceil((self.imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = self.classifier.getPrediction(imgWhite)
            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50), (x - self.offset + 225, y - self.offset), (255, 4, 255), cv2.FILLED)
            cv2.putText(imgOutput, self.labels[index], (x + 20, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (252, 4, 252), 4)

        return frame.from_ndarray(imgOutput, format="bgr24")

def main():
    st.title('ASL Gesture Recognition App')

    webrtc_ctx = webrtc_streamer(
        key="asl-recognition",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    if webrtc_ctx.video_processor:
        st.write("Camera is on.")
    else:
        st.write("Camera is off.")

if __name__ == '__main__':
    main()
