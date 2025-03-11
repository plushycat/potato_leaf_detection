import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os


file_id = "14dNmltgUuWh_S_H10ID38dgfakt9aVve"
url="https://drive.google.com/file/d/14dNmltgUuWh_S_H10ID38dgfakt9aVve/view?usp=sharing"
model_path = 'trained_plant_disease_model.keras'

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")    
    gdown.download(url,model_path,quiet=False)


def model_prediction(test_image):
    model=tf.keras.model(model_path)
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title('POTATO DISEASE DETECTION SYSTEM')
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])

if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)

elif(app_mode=="DISEASE RECOGNITION"):
    st.header('Plant disease detection')
    test_image=st.file_uploader('Choose an image:')
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))