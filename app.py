import streamlit as st
from pathlib import Path
from PIL import Image
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
import random

# Import OpenCV
import cv2

# Utility
import random

import itertools
import random
from collections import Counter
from glob import iglob

st.set_page_config(page_title="Plant Disease App")

# Set the directory path
my_path = "."

# test = pd.read_csv(my_path + "/data/sample.csv")
# img_1_path = my_path + "/images/img_1.jpg"
# img_2_path = my_path + "/images/img_2.jpg"
# img_3_path = my_path + "/images/img_3.jpg"
banner_path = my_path + "/images/banner.png"
# output_image = my_path + "/images/gradcam2.png"
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

saved_model_path = "C:/Users/yraj/Work/POCs/Dr. Greenry House/model"
model = tf.keras.models.load_model(saved_model_path)  # Loading Saved Model


# @st.cache
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


# ----------------------------STREAMLIT APP----------------------------

# Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)


# Creating Customized Sidebar

# Read and display the banner
activities = ["Assistant🤖", "About"]
choice = st.sidebar.selectbox("Select Activty", activities)
# st.sidebar.image(banner_path, use_column_width=True)

if choice == "About":
    intro_markdown = read_markdown_file("./doc/about.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
if choice == "Assistant🤖":
    # -------------------FUNCTIONS----------------------------------
    # for performing image ingestion/selection and producing the class of disease
    def classify_disease(img_loc: str):
        # Setup Image shape and batch size
        def load_image(filename):
            img = cv2.imread(img_loc)
            img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
            img = img / 255

            return img

        def predict(image):
            probabilities = model.predict(np.asarray([img]))[0]
            class_idx = np.argmax(probabilities)

            return {classes[class_idx]: probabilities[class_idx]}

        img = load_image(img_loc)
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
        )

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

    # Infering using the model
    result_print, disease_found = classify_disease(
        "C:/Users/yraj/Work/POCs/Dr. Greenry House/data/test/AppleCedarRust2.JPG"
    )

    # -----------------------APP-LAYOUT-----------------------
    APP_ICON_URL = (
        "https://i.pinimg.com/736x/4e/af/08/4eaf081c599286fd9ca84c1757c07152.jpg"
    )
    st.write(
        "<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>",
        unsafe_allow_html=True,
    )
    st.image(APP_ICON_URL, width=80)

    # Set App title
    st.title("Dr. Greenry House🥸")
    # App description
    st.write(
        "The app provides assistance on plant diseases, by identifying them providing information about them."
    )
    st.write(
        "**App is capable of providing assistance for 38 diseases for different plants**"
    )
    st.markdown("***")

    # Sidebar
    st.sidebar.write("**Select an image for a DEMO**")
    menu = ["Select an Image", "Image 1", "Image 2", "Image 3"]
    choice = st.sidebar.selectbox("Select an image", menu)

    st.sidebar.write("**Upload your Image**")
    upload_image = st.sidebar.file_uploader(
        "Upload your image in JPG or PNG format", type=["jpg", "png"]
    )

    if upload_image is not None:
        result_print, disease_found, img = classify_disease_uploaded_file(upload_image)
        st.markdown(result_print, unsafe_allow_html=True)
        st.image(img, channels="BGR")
        del upload_image

    elif choice == "Image 1":
        # Deploy the model if the user selects Image 1
        image_1_loc = "C:/Users/yraj/Work/POCs/Dr. Greenry House/data/test/TomatoYellowCurlVirus1.JPG"
        result_print, disease_found = classify_disease(image_1_loc)
        st.markdown(result_print, unsafe_allow_html=True)

        image_1 = Image.open(image_1_loc)
        st.image(image_1)
    elif choice == "Image 2":
        # Deploy the model if the user selects Image 2
        image_1_loc = (
            "C:/Users/yraj/Work/POCs/Dr. Greenry House/data/test/TomatoHealthy4.JPG"
        )
        result_print, disease_found = classify_disease(image_1_loc)
        st.markdown(result_print, unsafe_allow_html=True)

        image_1 = Image.open(image_1_loc)
        st.image(image_1)
    elif choice == "Image 3":
        # Deploy the model if the user selects Image 3
        image_1_loc = (
            "C:/Users/yraj/Work/POCs/Dr. Greenry House/data/test/TomatoEarlyBlight4.JPG"
        )
        result_print, disease_found = classify_disease(image_1_loc)
        st.markdown(result_print, unsafe_allow_html=True)

        image_1 = Image.open(image_1_loc)
        st.image(image_1)
    # prompt = st.text_input("Plug in your queries here!!")
    # # Set red flag if no image is selected/uploaded
    # if uploaded_image is None and choice == "Select an Image":
    #     st.sidebar.markdown(app_off, unsafe_allow_html=True)
    #     st.sidebar.markdown(app_off2, unsafe_allow_html=True)

    # -------------------------------LLM---------------------------------
    import os
    import streamlit as st
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

    llm = openai_llm

    # st.title("🦜🔗 YouTube GPT Creator")
    prompt = st.text_input(
        "For more information about the disease, enter it's name as you see above."
    )

    # Prompt templates
    title_template = PromptTemplate(
        input_variables=["topic"],
        template="Tell me about {topic} in breif",
    )

    script_template = PromptTemplate(
        input_variables=["title", "wikipedia_research"],
        template="write me precise steps to treat the disease: {title} while leveraging this wikipedia reserch:{wikipedia_research} ",
    )

    # Memory
    title_memory = ConversationBufferMemory(
        input_key="topic", memory_key="chat_history"
    )
    script_memory = ConversationBufferMemory(
        input_key="title", memory_key="chat_history"
    )

    # llm chains

    # Sequential Chain
    title_chain = LLMChain(
        llm=llm,
        prompt=title_template,
        verbose=True,
        output_key="title",
        memory=title_memory,
    )
    script_chain = LLMChain(
        llm=llm,
        prompt=script_template,
        verbose=True,
        output_key="script",
        memory=script_memory,
    )

    wiki = WikipediaAPIWrapper()

    # Show stuff to the screen if there's a prompt
    if prompt:
        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        script = script_chain.run(title=title, wikipedia_research=wiki_research)

        st.write(title)
        st.write(script)

        with st.expander("Title History"):
            st.info(title_memory.buffer)

        with st.expander("Script History"):
            st.info(script_memory.buffer)

        with st.expander("Wikipedia Research"):
            st.info(wiki_research)
