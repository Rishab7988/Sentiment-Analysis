<h1> Live Sentiment-Analysis with deployment on stream-lit</h1>
<br>
<h2> This project utilizes live webcam input to perform real-time sentiment analysis. By leveraging computer vision and natural language processing techniques, it identifies and analyzes facial expressions to determine the emotional state of the person in front of the camera.</h2>
<br>
<h2> Objective </h2>
<br>
1.Train a CNN model to detect faces and then classify the recieved face into 7 emotions (happy,sad,surprised,neutral,disgust,angry,fear).
<br>
2.Deploy the model using Streamlit. 
<br>
<h2> Tech-stack used </h2>
1. Tensorflow, keras, mobileNetv2 model for transfer learning(since I was new to transfer learning the kaggle community was really helpful)
2. OpenCV for image processing.
3. Streamlit_webrtc (so that webcam can be accessed from the net, and not limited to local system)
4. Streamlit for deployment
