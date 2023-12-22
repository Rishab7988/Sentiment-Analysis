import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

# OPENCV_LOG_LEVEL=0
st.set_page_config(layout="wide")
st.title("LIVE SENTIMENT ANALYSIS 😄 😡 😞 😲 🤢 😨 😐 ...")

frame_placeholder=st.empty()
stop_button=st.button("STOP")

load_model=tf.keras.models.load_model(r'final_model.h5')



path=r'haarcascade_frontalface_default.xml'
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN


#set rectangular background to white
rectangle_bgr=(225,225,225)
#make a black image
img=np.zeros((500,500))

text="Some text in a box!"

(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]

text_offset_x=10
text_offset_y=img.shape[0]-25

#make the coords of the box with a small padding of two pixels
box_coords=((text_offset_x,text_offset_y),(text_offset_x + text_width + 2, text_offset_y - text_height -2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)


cap=cv2.VideoCapture(0)

# if not cap.isOpened():
#     cap=cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

while cap.isOpened() and not stop_button:
    ret,frame=cap.read()

    if not ret:
       st.write("Video capture has ended.")
       break
    

   #  frame_placeholder.image(frame,channels="BGR")
    
    faceCascade=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    

    face_roi=frame

    faces=faceCascade.detectMultiScale(gray,1.1,4)
    
    for x,y,w,h in faces:
      roi_gray=gray[y:y+h, x:x+w]
      roi_color=frame[y:y+h, x:x+w]
      cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

      
      facess=faceCascade.detectMultiScale(roi_gray)
      if(len(facess)==0):
         print("Face not detected")
      else:
         for(ex,ey,ew,eh) in facess:
            face_roi=roi_color[ey:ey+eh, ex:ex + ew]

    final_image=cv2.resize(face_roi, (224,224)) # resizing the image to fit it in the model
    final_image=np.expand_dims(final_image,axis=0) # need the fourth dimension
    final_image=final_image/255.0 #normalizing the data
    



    font=cv2.FONT_HERSHEY_SIMPLEX

    predictions=load_model.predict(final_image)

    font_scale=1.5
    font=cv2.FONT_HERSHEY_PLAIN


    if(np.argmax(predictions)==0):
       status="Angry"

       x1,y1,w1,h1=0,0,175,75

       frame=cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

       frame=cv2.putText(frame,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

       frame=cv2.putText(frame,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

       frame=cv2.rectangle(frame,(x1,y1), (x1+w1, y1+h1), (0,0,255))

       


    elif(np.argmax(predictions)==1):
       status="Disgust"

       x1,y1,w1,h1=0,0,175,75

       frame=cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

       frame=cv2.putText(frame,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

       frame=cv2.putText(frame,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

       frame=cv2.rectangle(frame,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

    elif(np.argmax(predictions)==2):
       status="Fear"

       x1,y1,w1,h1=0,0,175,75

       frame=cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

       frame=cv2.putText(frame,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

       frame=cv2.putText(frame,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

       frame=cv2.rectangle(frame,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

    elif(np.argmax(predictions)==3):
       
       
       status="Happy"

       x1,y1,w1,h1=0,0,175,75

       frame=cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

       frame=cv2.putText(frame,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

       frame=cv2.putText(frame,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

       frame=cv2.rectangle(frame,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

    elif(np.argmax(predictions)==4):
       status="Sad" 

       x1,y1,w1,h1=0,0,175,75

       frame=cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

       frame=cv2.putText(frame,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

       frame=cv2.putText(frame,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

       frame=cv2.rectangle(frame,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

    elif(np.argmax(predictions)==5):
       status="Neutral"

       x1,y1,w1,h1=0,0,175,75

       frame=cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

       frame=cv2.putText(frame,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

       frame=cv2.putText(frame,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

       frame=cv2.rectangle(frame,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       

    elif(np.argmax(predictions)==6):
       status="Surprised"

       x1,y1,w1,h1=0,0,175,75

       frame=cv2.rectangle(frame, (x1,x1), (x1 + w1, y1 + h1), (0,0,0), -1)

       frame=cv2.putText(frame,status,(x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

       frame=cv2.putText(frame,status,(100,150), font, 3, (0,0,255),2,cv2.LINE_4)

       frame=cv2.rectangle(frame,(x1,y1), (x1+w1, y1+h1), (0,0,255))    

       
    frame_placeholder.image(frame, channels="BGR")

   #  cv2.imshow('Face Emotion Recognition',frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
       break

# while True:
    

cap.release()
cv2.destroyAllWindows()
    
    
    



