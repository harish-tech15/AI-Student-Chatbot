import os
import json
import pickle
import random

import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Student Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "chatbot_model.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tokenizer.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "label_encoder.pkl"
)

INTENTS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "intents.json"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_SEQUENCE_LENGTH = 6
CONFIDENCE_THRESHOLD = 0.35


# ============================================================
# NORMALIZE TAG
# ============================================================

def normalize_tag(tag):
    return (
        str(tag)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# ============================================================
# LOAD CHATBOT
# ============================================================

@st.cache_resource
def load_chatbot():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}"
        )

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            f"Label encoder file not found: {LABEL_ENCODER_PATH}"
        )

    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(
            f"Intents file not found: {INTENTS_PATH}"
        )

    # Load model
    model = keras.saving.load_model(
        MODEL_PATH,
        compile=False
    )

    # Load tokenizer
    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:
        tokenizer = pickle.load(file)

    # Load label encoder
    with open(
        LABEL_ENCODER_PATH,
        "rb"
    ) as file:
        label_encoder = pickle.load(file)

    # Load intents
    with open(
        INTENTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        intents = json.load(file)

    return (
        model,
        tokenizer,
        label_encoder,
        intents
    )


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(
    user_text,
    model,
    tokenizer,
    label_encoder
):

    # Convert text to numbers
    sequence = tokenizer.texts_to_sequences(
        [user_text]
    )

    # Pad to model input size
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # Predict
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    # Predicted class
    predicted_index = int(
        np.argmax(prediction[0])
    )

    # Confidence
    confidence = float(
        np.max(prediction[0])
    )

    # Class number -> intent name
    intent_name = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return (
        intent_name,
        confidence,
        padded_sequence
    )


# ============================================================
# GET RESPONSE
# ============================================================

def get_response(
    intent_name,
    intents
):

    predicted_tag = normalize_tag(
        intent_name
    )

    for intent in intents.get(
        "intents",
        []
    ):

        json_tag = normalize_tag(
            intent.get("tag", "")
        )

        if json_tag == predicted_tag:

            responses = intent.get(
                "responses",
                []
            )

            if len(responses) > 0:
                return random.choice(
                    responses
                )

    return (
        "Sorry, I don't have an answer "
        "for that question yet. "
        "Please try asking another question."
    )


# ============================================================
# LOAD FILES
# ============================================================

try:

    (
        model,
        tokenizer,
        label_encoder,
        intents
    ) = load_chatbot()

except Exception as error:

    st.error(
        "❌ Chatbot could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Student Chatbot")

st.write(
    "Ask questions about AI, Machine Learning, "
    "Deep Learning, Python, NLP and other "
    "student topics."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 About")

    st.write(
        "This chatbot uses a Deep Learning "
        "LSTM model to classify student "
        "questions and generate responses."
    )

    st.divider()

    st.subheader("🛠️ Technologies")

    st.write(
        """
        • Python
        • TensorFlow
        • Keras
        • LSTM
        • NLP
        • Streamlit
        """
    )

    st.divider()

    st.write(
        f"Model Input Length: {MAX_SEQUENCE_LENGTH}"
    )

    st.write(
        f"Confidence Threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask your question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_input:

    # User message
    with st.chat_message("user"):

        st.markdown(
            user_input
        )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Default values
    predicted_intent = "Unknown"
    confidence = 0.0
    input_sequence = []

    try:

        # Predict intent
        (
            predicted_intent,
            confidence,
            input_sequence
        ) = predict_intent(
            user_input,
            model,
            tokenizer,
            label_encoder
        )

        # Generate response
        if confidence >= CONFIDENCE_THRESHOLD:

            response = get_response(
                predicted_intent,
                intents
            )

        else:

            response = (
                "I'm not completely sure about "
                "that question. Please try asking "
                "it in another way."
            )

    except Exception as error:

        response = (
            "Sorry, something went wrong "
            "while processing your question."
        )

        st.error(
            f"Error: {error}"
        )

    # Assistant response
    with st.chat_message("assistant"):

        st.markdown(
            response
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # Prediction details
    with st.expander(
        "🔎 Prediction Details"
    ):

        st.write(
            f"**Predicted Intent:** "
            f"{predicted_intent}"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence * 100:.2f}%"
        )

        st.write(
            "**Input Tokens:**"
        )

        if hasattr(
            input_sequence,
            "tolist"
        ):

            st.write(
                input_sequence.tolist()
            )

        else:

            st.write(
                input_sequence
            )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader(
    "💡 Example Questions"
)

st.info(
    "What is Machine Learning?"
)

st.info(
    "What is Deep Learning?"
)

st.info(
    "What is Artificial Intelligence?"
)

st.info(
    "What is NLP?")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python, TensorFlow, Keras, "
    "LSTM and Streamlit"
)
