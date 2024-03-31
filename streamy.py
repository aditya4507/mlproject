import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from twilio.rest import Client

# Define image dimensions
img_height = 250
img_width = 250

def load_model():
    try:
        # Load the pre-trained model
        model = tf.keras.models.load_model("/Users/adityatapase/Desktop/major zip/Model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

# Adjust the threshold and handle false detections better
def predict_frame(img):
    if model is None:
        return "Model not loaded"
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    threshold = 0.5  # Adjust this threshold as needed
    if prediction[0][0] > threshold:
        return "Accident Detected"
    else:
        return "No Accident"

# Main Streamlit app
def main():
    



    st.title("Accident Detection From Video Using Deep Learning ")

    # Frame number input
    selected_frame_number = st.text_input("Enter frame number (1 to 75)", "")

    try:
        selected_frame = int(selected_frame_number)
        if not 1 <= selected_frame <= 75:
            st.warning("Frame number should be between 1 and 75.")
            return
    except ValueError:
        st.error("Please enter a valid integer.")
        return

    # File uploader for video file
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Perform prediction
        cap = cv2.VideoCapture(temp_file_path)
        c = 1
        image = []
        label = []
        accident_detected = False
        while True:
            grabbed, frame = cap.read()
            if c % 10 == 0:  # Adjust the frame sampling rate as needed
                resized_frame = tf.image.resize(frame, (img_height, img_width))  # Resize frame using TensorFlow
                image.append(frame)
                label.append(predict_frame(resized_frame))
                if label[-1] == "Accident Detected":
                    accident_detected = True
            c += 1
            if not grabbed:
                break
        cap.release()

        # Display the selected frame
        if label[selected_frame - 1] == "Accident Detected":
            st.image(image[selected_frame - 1], caption=f"Frame: {label[selected_frame - 1]}", use_column_width=True)
            # Send alert if accident detected
            account_sid = "ACd1d7b77c40a4382788ae1ca77de4a9a1"
            auth_token = "b5e76b73435bf0d2d3f9aebb5aca7254"
            client = Client(account_sid, auth_token)

            # Making a call using Twilio
            call = client.calls.create(
                url="http://demo.twilio.com/docs/voice.xml",
                to="+918208126186",
                from_="+17074666767"
            )

            # Sending a message using Twilio
            message = client.messages.create(
                from_='+17074666767',
                body='Hi The car met with an accident by Deeshant',
                to='+918208126186'
            )

            st.success("Accident Detected! Alert Sent.")
        else:
            st.image(image[selected_frame - 1], caption=f"Frame: {label[selected_frame - 1]}", use_column_width=True)

if __name__ == "__main__":
    main()
 