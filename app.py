import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

# OR reduce watching scope
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import numpy as np
import streamlit as st
from PIL import Image
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Load VGG16 model for feature extraction
@st.cache_resource
def load_feature_extractor():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

# Load trained classifier
@st.cache_resource
def load_classifier():
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)

# Feature extraction
def extract_features(img, feature_model):
    img = img.resize((128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_model.predict(img)
    return features.reshape(1, -1)

# Show homepage
def show_home():
    st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

# Title and header
st.title(" Brain Tumor Detection ")
st.subheader("Detects Tumor Presence from MRI Scans")

st.markdown("""
        ## What is a Brain Tumor?
        A brain tumor is an abnormal growth of cells in the brain. It can be **malignant (cancerous)** or **benign (non-cancerous)**. Tumors affect brain function and can cause serious complications if untreated.

        ### Types of Tumors:
        - **Glioma**: Originates in the glial cells. Often aggressive.
        - **Meningioma**: Develops in the meninges, typically benign but large ones can be harmful.
        - **Pituitary Tumor**: Forms in the pituitary gland, may disrupt hormone production.
        - **No Tumor**: Normal brain MRI.

        ### Common Symptoms:
        - Headaches (especially in the morning)
        - Nausea or vomiting
        - Vision problems
        - Seizures
        - Difficulty with balance or speech

        ---
        Use the menu to navigate through prediction or self-assessment.
    """)

# Show prediction page
def show_prediction_page(feature_model, clf):
    st.title("🔍 Class-wise Brain Tumor detection")

    uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Tumor Type"):
            features = extract_features(image, feature_model)
            prediction = clf.predict(features)
            prob = clf.predict_proba(features)[0]

            labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            st.success(f"📌 Prediction: **{labels[prediction[0]]}**")

            st.write("### Prediction Probabilities:")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.pie(
            prob,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90
            )
            ax.axis('equal')  # Equal aspect ratio for a perfect circle
            st.pyplot(fig)


# Show self-assessment
def show_self_assessment():
    st.title("🩺 Self Assessment")

    st.markdown("Answer the following to get a basic risk score:")

    q1 = st.selectbox("Do you experience frequent or severe headaches?", ["No", "Yes",])
    q2 = st.selectbox("Do you have blurred vision or vision loss?", ["No", "Yes"])
    q3 = st.selectbox("Do you feel dizzy or have balance issues?", ["No", "Yes"])
    q4 = st.selectbox("Have you experienced seizures recently?", ["No", "Yes"])
    q5 = st.selectbox("Do you have unexplained nausea or vomiting?", ["No", "Yes"])

    if st.button("Assess My Risk"):
        score = sum([q == "Yes" for q in [q1, q2, q3, q4, q5]])
        if score >= 4:
            st.error("⚠️ High Risk. Please consult a neurologist immediately.")
        elif score >= 2:
            st.warning("⚠️ Moderate Risk. Consider medical evaluation.")
        else:
            st.success("✅ Low Risk. You seem to be okay, but stay aware of symptoms.")
# Load models
feature_model = load_feature_extractor()
clf = load_classifier()

# Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Predict Tumor", "Self Assessment"])

if app_mode == "Home":
    show_home()
elif app_mode == "Predict Tumor":
    show_prediction_page(feature_model, clf)
elif app_mode == "Self Assessment":
    show_self_assessment()


