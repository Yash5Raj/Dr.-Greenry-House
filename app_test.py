# ------------------------------IMPORTING-REQUIRED-LIBRARIES---------------------
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message

from pathlib import Path
from PIL import Image
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
import random

# Import OpenCV
import cv2

# LLM Model required libraries
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import (
    SimpleSequentialChain,
    SequentialChain,
)  # for allowing multiple sets of output
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# using OpenAI
from llm_models import openai_llm

# -----------------------------------ML-Models----------------------------------

# -----------------------------------Computer-Vision----------------------------
# Set the directory path
my_path = "."
banner_path = my_path + "/images/banner.png"

IMAGE_SHAPE = (224, 224)

BATCH_SIZE = 64  # @param {type:"integer"}

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def classify_disease_uploaded_file(upload_image):
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    def load_image(opencv_image):
        img = cv2.resize(opencv_image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        img = img / 255

        return img

    def predict(image):
        probabilities = model.predict(np.asarray([img]))[0]
        class_idx = np.argmax(probabilities)

        return {classes[class_idx]: probabilities[class_idx]}

    img = load_image(opencv_image)
    prediction = predict(img)
    disease_found = list(prediction.keys())[0].replace("_", " ")
    img_location = (
        "C:/Users/yraj/Work/POCs/Dr. Greenry House/output/success_{0}.jpg".format(
            list(prediction.keys())[0]
        )
    )

    return (
        "PREDICTED Class: %s, Confidence: %f"
        % (
            list(prediction.keys())[0].replace("_", " "),
            list(prediction.values())[0],
        ),
        disease_found,
        img,
    )


# Loading Saved Model
saved_model_path = "C:/Users/yraj/Work/POCs/Dr. Greenry House/model"
model = tf.keras.models.load_model(saved_model_path)

# --------------------------------------LLM-Model-------------------------------------

llm = openai_llm


def image_upload_response(disease_found):
    cure_template = PromptTemplate(
        input_variables=["disease_found"],
        template="tell me about disease {disease_found} in breif, and provide a step-by-step possible treatment for it.",
    )
    # Memory
    cure_memory = ConversationBufferMemory(
        input_key="disease_found", memory_key="chat_history"
    )
    # Sequential Chain
    cure_chain = LLMChain(
        llm=llm,
        prompt=cure_template,
        verbose=True,
        output_key="title",
        memory=cure_memory,
    )

    wiki = WikipediaAPIWrapper()

    wiki_research = wiki.run(disease_found)

    return cure_chain.run(disease_found)


# -------------------------------------STREAMLIT-APP-----------------------------------

st.set_page_config(page_title="Plant Care Assistant")
# Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

APP_ICON_URL = "https://i.pinimg.com/736x/4e/af/08/4eaf081c599286fd9ca84c1757c07152.jpg"
st.write(
    "<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>",
    unsafe_allow_html=True,
)
st.image(APP_ICON_URL, width=80)

# Sidebar for App Overview
with st.sidebar:
    st.title("Plant Disease Detection App with Assistant🤖")
    st.markdown(
        """
    ## About
    This app is an LLM-powered Assistant built using:
    - [Streamlit](https://streamlit.io/)
    - [OpenAI](https://openai.com/)
    - And Much More to be added
    """
    )
    add_vertical_space(5)
    st.markdown(
        """
        - [GitHub Repo](https://github.com/YashRaj5/Dr._Greenry_House/tree/main)
        """
    )
    st.write(
        "Made with ❤️ by [Yash Raj](https://in.linkedin.com/in/yash-raj-2841641b3)"
    )

# Set App title
st.title("Dr. Greenry House🥸")

# For uploading Image
st.write("**Upload your Image**")
upload_image = st.file_uploader(
    "Upload image of plant in JPG or PNG format", type=["jpg", "png"]
)

# Generate empty lists for generated and past.
## generated stores AI generated responses
if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "Hello, I am Dr. Greenry, Please upload the image of the infected plant!"
    ]
## past stores User's questions
# if "past" not in st.session_state:
#     st.session_state["past"] = ["Hello"]

# Layout of input/response containers
input_container = st.container()
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


## Applying the user input box
with input_container:
    user_input = get_text()


# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def user_input_response(prompt):
    response = llm.run(prompt)
    return response


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if st.session_state["generated"]:
        if upload_image is not None:
            # Performing Inference on the Image
            result_print, disease_found, img = classify_disease_uploaded_file(
                upload_image
            )
            st.markdown(result_print, unsafe_allow_html=True)
            st.image(img, channels="BGR")
            output = image_upload_response(disease_found)
            st.session_state.generated.append(output)
            message(st.session_state["generated"][-1], key=str(-1))
            del upload_image
        else:
            message(st.session_state["generated"][0], key=str(0))
