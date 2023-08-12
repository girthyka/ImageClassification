# Save the Streamlit app code to "my_app.py" file
# You don't need the "%%writefile my_app.py" line in the script when running it in a local Python environment.

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


# Set a title for your Streamlit app
st.title("Image Emotion Classifier")

# Define the sidebar components
upload_file = st.sidebar.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])
generate_pred = st.sidebar.button("Predict")

# Load your trained model
model = tf.keras.models.load_model('imageclassifier.h5')

# Function to preprocess the image and make predictions
def import_n_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis, ...]
    pred = model.predict(reshape)
    return pred

# When the "Predict" button is clicked
if generate_pred:
    if upload_file is not None:
        image = Image.open(upload_file)
        
        # Display the uploaded image
        with st.expander('Uploaded Image', expanded=True):
            st.image(image, use_column_width=True)
        
        # Get predictions using your model
        pred = import_n_predict(image, model)
        
        labels = ['happyyy', 'sad:(']
        predicted_label = labels[np.argmax(pred)]
        
        # Display the prediction
        st.title(f"The photo looks {predicted_label}")

# Run the Streamlit app
if __name__ == '__main__':
    st.write("To use this app, please upload an image and click 'Predict' in the sidebar.")
