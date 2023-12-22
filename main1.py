import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av

st.title("LIVE SENTIMENT ANALYSIS 😄 😡 😞 😲 🤢 😨 😐")
load_model=tf.keras.models.load_model(r'final_model.h5')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Happy', 3: 'Sad', 4: 'Neutral', 5: 'Surprised'}

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(image=img_gray, scaleFactor=1.1, minNeighbors=3)

        face_roi=img

        for (x, y, w, h) in faces:
                cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=3)
                roi_gray=img_gray[y:y+h, x:x+w]
                roi_color=img[y:y+h, x:x+w]
                
                facess=faceCascade.detectMultiScale(roi_gray)
                if(len(facess)==0):
                        print("Face not detected")
                else:
                        for(ex,ey,ew,eh) in facess:
                                face_roi=roi_color[ey:ey+eh, ex:ex + ew]

        final_image=cv2.resize(face_roi, (224,224)) # resizing the image to fit it in the model
        final_image=np.expand_dims(final_image,axis=0) # need the fourth dimension
        final_image=final_image/255.0 #normalizing the data


        predictions=load_model.predict(final_image)


        if(np.argmax(predictions)==0):
                status="Angry"

                x1,y1,w1,h1=0,0,175,75

                cv2.rectangle(img, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

                cv2.putText(img,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(img,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)
                
                cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1), (0,0,255))


        elif(np.argmax(predictions)==1):
                status="Disgust"

                x1,y1,w1,h1=0,0,175,75

                cv2.rectangle(img, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

                cv2.putText(img,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(img,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

                cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

        elif(np.argmax(predictions)==2):
                status="Fear"

                x1,y1,w1,h1=0,0,175,75

                cv2.rectangle(img, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

                cv2.putText(img,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(img,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

                cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

        elif(np.argmax(predictions)==3):
                status="Happy"

                x1,y1,w1,h1=0,0,175,75

                cv2.rectangle(img, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

                cv2.putText(img,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(img,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

                cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

        elif(np.argmax(predictions)==4):
                status="Sad" 

                x1,y1,w1,h1=0,0,175,75

                cv2.rectangle(img, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

                cv2.putText(img,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(img,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

                cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

        elif(np.argmax(predictions)==5):
                status="Neutral"

                x1,y1,w1,h1=0,0,175,75

                cv2.rectangle(img, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

                cv2.putText(img,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(img,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

                cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

        elif(np.argmax(predictions)==6):
                status="Surprised"

                x1,y1,w1,h1=0,0,175,75

                cv2.rectangle(img, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

                cv2.putText(img,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(img,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

                cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1), (0,0,255))
        
        return av.VideoFrame.from_ndarray(img,format="bgr24")

def main():
    st.write("Select preferred input device and click Start.")
    webrtc_streamer(key="key", rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor)

    html_temp4 = """
                        <div style="padding:10px">
                        <h4 style="color:white;text-align:center;">Developed by Rishab Mahato using OpenCV,Streamlit,Tensorflow and Keras.</h4>
                        <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                        </div>
                        <a href="https://github.com/Rishab7988">Github</a>
                        <a href="https://www.linkedin.com/in/rishab-mahato/">LinkedIn</a>
                        <br></br>
                        <br></br>"""

    st.markdown(html_temp4, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
