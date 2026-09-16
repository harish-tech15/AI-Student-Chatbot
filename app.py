
import os
import json
import pickle

import numpy as np
import streamlit as st
import keras


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Student Chatbot",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "model/chatbot_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
INTENTS_PATH = "data/intents.json"


# --------------------------------------------------
# LOAD CHATBOT
# --------------------------------------------------

@st.cache_resource
def load_chatbot():

    # Check model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Check tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}"
        )

    # Check label encoder
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            f"Label encoder file not found: {LABEL_ENCODER_PATH}"
        )

    # Check intents
    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(
            f"Intents file not found: {INTENTS_PATH}"
        )

    # ----------------------------------------------
    # Load Keras model
    # ----------------------------------------------

    model = keras.saving.load_model(
        MODEL_PATH,
        compile=False
    )

    # ----------------------------------------------
    # Load tokenizer
    # ----------------------------------------------

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    # ----------------------------------------------
    # Load label encoder
    # ----------------------------------------------

    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    # ----------------------------------------------
    # Load intents
    # ----------------------------------------------

    with open(INTENTS_PATH, "r", encoding="utf-8") as file:
        intents = json.load(file)

    return model, tokenizer, label_encoder, intents


# --------------------------------------------------
# LOAD MODEL SAFELY
# --------------------------------------------------

try:

    model, tokenizer, label_encoder, intents = load_chatbot()

except Exception as e:

    st.error("❌ Chatbot model could not be loaded.")

    st.code(str(e))

    st.info(
        "Please check the model, tokenizer, label encoder, "
        "intents.json and requirements.txt files."
    )

    st.stop()


# --------------------------------------------------
# FIND INTENT RESPONSE
# --------------------------------------------------

def get_response(intent_name):

    """
    Find the response corresponding to the predicted intent.
    """

    # Case 1: intents.json is a dictionary
    if isinstance(intents, dict):

        # Standard format:
        # {
        #   "intents": [
        #       {
        #           "tag": "...",
        #           "patterns": [],
        #           "responses": []
        #       }
        #   ]
        # }

        intent_list = intents.get("intents", [])

        for intent in intent_list:

            if intent.get("tag") == intent_name:

                responses = intent.get("responses", [])

                if responses:
                    return responses[0]

    return "Sorry, I don't have an answer for that question yet."


# --------------------------------------------------
# PREDICT INTENT
# --------------------------------------------------

def predict_intent(user_text):

    """
    Convert user text into a sequence and predict intent.
    """

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([user_text])

    # Convert to numpy array
    sequence = np.array(sequence)

    # Model prediction
    prediction = model.predict(
        sequence,
        verbose=0
    )

    # Get highest probability
    predicted_index = int(np.argmax(prediction[0]))

    confidence = float(
        np.max(prediction[0])
    )

    # Convert index to intent name
    try:

        intent_name = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

    except Exception:

        intent_name = str(predicted_index)

    return intent_name, confidence


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 AI Student Chatbot")

st.write(
    "Ask questions about Machine Learning, "
    "Deep Learning, Python, AI and Data Science."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Student Assistant")

    st.write(
        "This chatbot uses a trained Deep Learning "
        "model to identify the user's intent."
    )

    st.divider()

    st.subheader("Example Questions")

    st.write("• What is Machine Learning?")

    st.write("• What is Deep Learning?")

    st.write("• What is Python?")

    st.write("• What is Artificial Intelligence?")

    st.write("• What is Data Science?")

    st.divider()

    st.caption(
        "AI Student Chatbot"
    )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

user_input = st.chat_input(
    "Ask your question..."
)


# --------------------------------------------------
# CHATBOT RESPONSE
# --------------------------------------------------

if user_input:

    # Display user message
    with st.chat_message("user"):

        st.write(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Predict
    try:

        intent_name, confidence = predict_intent(
            user_input
        )

        response = get_response(
            intent_name
        )

        # Confidence threshold
        if confidence < 0.35:

            response = (
                "I'm not fully sure about that question. "
                "Please try asking it in a different way."
            )

    except Exception as e:

        response = (
            "Sorry, something went wrong while "
            "processing your question."
        )

        st.error(str(e))

    # Display bot response
    with st.chat_message("assistant"):

        st.write(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python, TensorFlow/Keras and Streamlit"
)
