import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

st.set_page_config(layout="wide")
st.title("LIVE SENTIMENT ANALYSIS 😄 😡 😞 😲 🤢 😨 😐")
load_model = tf.keras.models.load_model('final_model.h5')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Happy', 3: 'Sad', 4: 'Neutral', 5: 'Surprised'}

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoTransformer(VideoProcessorBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = load_model.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

def main():
    st.write("Click Start")
    webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                                 video_processor_factory=VideoTransformer)
    frame_placeholder = st.empty()
    
    if webrtc_ctx:
        while True:
            video_frame = webrtc_ctx.video_transformer.recv()
            if video_frame is None:
                break

            processed_frame = video_frame.to_ndarray(format="bgr24")
            detector = VideoTransformer()
            processed_frame = detector.transform(video_frame)

            frame_placeholder.image(processed_frame, channels="BGR")

if __name__ == "__main__":
    main()
