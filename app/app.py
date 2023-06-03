from os import path, remove
from random import randint

import streamlit as st
from PIL import Image

from model import get_model, get_model_summary, model_prediction

#from model import get_model, get_model_summary, model_prediction

st.set_page_config(page_title="Deepfake Image Detector",
                   page_icon=path.join('assets', 'icons', 'logo-5.png'))

with open(path.join('assets', 'styles.css')) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

_, page_banner_img, _ = st.columns([3, 2, 3])
page_banner_img.image(path.join('assets', 'icons', 'logo-4.png'),
                      use_column_width=True)

#st.title('Deepfake Image Detector ')
st.subheader('')

#st.markdown()

model_cnn = get_model(path.join('assets', 'model', 'model.hdf5'))
model_summary = get_model_summary(model_cnn)

# Print model summary to expandable container
#with st.expander(label='Model architecture summary', expanded=False):
 #st.markdown(f"""```{model_summary}""")

# Upload face image file to use for prediction
uploaded_file = st.file_uploader('Upload a face image for prediction',
                                 type=['png', 'jpg', 'jpeg'],
                                 accept_multiple_files=False,
                                 help='Please select an image for prediction')

if uploaded_file is not None and uploaded_file.type.split('/')[0] == 'image':

    uploaded_file_name = uploaded_file.name
    uploaded_image = Image.open(uploaded_file).save(uploaded_file_name)
    
    img = Image.open(uploaded_file_name).resize((480,480))
    
    col1, col2, col3 = st.columns(3)

    col1.write("")
    col2.image(img, use_column_width=True)
    col3.write("")

    prediction_label, real_face_prob, fake_face_prob = model_prediction(uploaded_file_name, model_cnn)
    remove(uploaded_file_name)  # scrap the file after prediction

else:

    #embedded_img = path.join('assets', 'faces', f'{randint(1, 6)}.jpg')
    prediction_label, real_face_prob, fake_face_prob = 'No Image Uploaded',0,0

with st.expander(label='Prediction Probabilities', expanded=False):
    st.markdown('<h3>Results</h3>', unsafe_allow_html=True)

    row_11, row_12 = st.columns([1, 1])
    row_11.markdown('<h5>Face Type</h5>', unsafe_allow_html=True)
    row_12.markdown('<h5>Probability</h5>', unsafe_allow_html=True)
    #st.progress(0)

    row_21, row_22 = st.columns([1, 1])
    row_21.markdown('<h3>REAL</h3>', unsafe_allow_html=True)
    row_22.markdown(f'<h2>{real_face_prob} %</h2>', unsafe_allow_html=True)
    st.progress(real_face_prob)

    row_31, row_32 = st.columns([1, 1])
    row_31.markdown('<h3>FAKE</h3>', unsafe_allow_html=True)
    row_32.markdown(f'<h2>{fake_face_prob} %</h2>', unsafe_allow_html=True)
    st.progress(fake_face_prob)

st.markdown(f"The classifier's prediction is that the loaded image is") 
st.markdown(f'<h4>{prediction_label}</h4>',unsafe_allow_html=True)

