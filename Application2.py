#libraries importation
import numpy as np
import streamlit as st 
from tensorflow.keras.models import load_model
import cv2


#load the model
tuned_model = load_model('C:/Users/840 G3/Desktop/M.Sc dissertation/dissertation_implementation/testing/optimized_model6.h5')

#dictionary to store each label
addiction_dict = {
    0 : 'Non-Addict',
    1 : 'Addict',
    }

#title of the application
st.header('Drug Addiction Classification System')

# Create a radio button to let the user choose
choice = st.radio("How would you like to provide an image?", ["Upload", "Take a picture"])

if choice == "Upload":
    image = st.file_uploader("Upload your Image ...", type=["jpg", "png"])
elif choice == "Take a picture":
    image = st.camera_input('Take a picture')


#set medium for image upload
#image = st.file_uploader("Upload your Image ...", type=["jpg", "png"])
#submit = st.button('Predict')
#image = st.camera_input('Take a picture')
submit = st.button('Predict')


#prediction
if submit:
    #check if image uploaded isn't null
    if image is not None:
        #convert file to opencv image
        file = np.asarray(bytearray(image.read()), dtype = np.uint8)
        cv_image = cv2.imdecode(file, 1)
        
        #display the image
        st.image(cv_image, channels="BGR")
        st.write(cv_image.shape)
        
        #resizing the image
        cv_image = cv2.resize(cv_image, (200, 200))
        
        #convert image to 4 dimension
        cv_image.shape = (1, 200, 200, 3)
        
        #make prediction
        y_pred = tuned_model.predict(cv_image)
        result = np.argmax(y_pred)
        
        #display result
        if result in addiction_dict:
            st.info(addiction_dict[result])
        else:
            st.info('I have no clue about this patient')