# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
# import cv2 as cv
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math

# # Initialize HandDetector and Classifier
# detector = HandDetector(maxHands=1)
# classifier = Classifier("keras_model.h5", "labels.txt")

# # Constants for image processing
# offset = 20
# imgSize = 300

# # Labels for classification
# labels = ["A", "B", "C", "Calm down", "D", "E", "F", "G", "H", "Hello", "I", "I hate you", "I love you", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "Stop", "T", "U", "V", "W", "X", "Y"]

# # VideoTransformer class for processing video frames
# class VideoTransformer(VideoTransformerBase):
#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         imgOutput = img.copy()  # Copy the frame for output
#         hands, img = detector.findHands(img)  # Detect hands in the frame
#         if hands:
#             hand = hands[0]  # Get the first detected hand
#             x, y, w, h = hand['bbox']  # Get the bounding box of the hand

#             # Create a white image for processing
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the hand region

#             # Calculate aspect ratio
#             aspectRatio = h / w

#             # If height is greater than width
#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv.resize(imgCrop, (wCal, imgSize))
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap:wCal + wGap] = imgResize

#             # If width is greater than height
#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv.resize(imgCrop, (imgSize, hCal))
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize

#             # Get prediction from the classifier
#             prediction, index = classifier.getPrediction(imgWhite)

#             # Draw a rectangle and label for the gesture
#             cv.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 225, y - offset), (255, 4, 255), cv.FILLED)
#             cv.putText(imgOutput, labels[index], (x + 20, y - 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             cv.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (252, 4, 252), 4)

#         return imgOutput

# def main():
#     st.title('ASL Gesture Recognition App')  # Title of the app

#     webrtc_ctx = webrtc_streamer(
#         key="example",
#         mode=WebRtcMode.SENDRECV,
#         video_transformer_factory=VideoTransformer,
#         media_stream_constraints={"video": True, "audio": False},  # No audio
#     )

# if __name__ == '__main__':
#     main()







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
    )

    if webrtc_ctx.video_processor:
        st.write("Camera is on.")
    else:
        st.write("Camera is off.")

if __name__ == '__main__':
    main()
