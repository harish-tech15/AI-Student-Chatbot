import os
import json
import pickle
import random

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Student Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# PROJECT PATHS
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
# CONSTANTS
# ============================================================

MAX_SEQUENCE_LENGTH = 6
CONFIDENCE_THRESHOLD = 0.35


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_tag(tag):
    """
    Normalize intent tags so that:
    machine learning
    Machine Learning
    machine-learning
    machine_learning

    are treated as the same tag.
    """

    return (
        str(tag)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# ============================================================
# LOAD CHATBOT FILES
# ============================================================

@st.cache_resource
def load_chatbot():

    # --------------------------------------------------------
    # Check Model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Check Tokenizer
    # --------------------------------------------------------

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file not found:\n{TOKENIZER_PATH}"
        )

    # --------------------------------------------------------
    # Check Label Encoder
    # --------------------------------------------------------

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            f"Label encoder file not found:\n{LABEL_ENCODER_PATH}"
        )

    # --------------------------------------------------------
    # Check Intents JSON
    # --------------------------------------------------------

    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(
            f"Intents file not found:\n{INTENTS_PATH}"
        )

    # --------------------------------------------------------
    # Load Keras Model
    # --------------------------------------------------------

    model = keras.saving.load_model(
        MODEL_PATH,
        compile=False
    )

    # --------------------------------------------------------
    # Load Tokenizer
    # --------------------------------------------------------

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    # --------------------------------------------------------
    # Load Label Encoder
    # --------------------------------------------------------

    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    # --------------------------------------------------------
    # Load Intents JSON
    # --------------------------------------------------------

    try:

        with open(
            INTENTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            intents = json.load(file)

    except json.JSONDecodeError as e:

        raise ValueError(
            "intents.json is invalid JSON.\n\n"
            f"Error: {e}"
        )

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

    # Convert text into sequence
    sequence = tokenizer.texts_to_sequences(
        [user_text]
    )

    # Pad sequence to model's expected input length
    sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # Model prediction
    prediction = model.predict(
        sequence,
        verbose=0
    )

    # Get predicted class
    predicted_index = int(
        np.argmax(prediction[0])
    )

    # Get confidence
    confidence = float(
        np.max(prediction[0])
    )

    # Convert class number into intent name
    intent_name = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return (
        intent_name,
        confidence,
        sequence
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

    # Search for matching intent
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

            if responses:

                return random.choice(
                    responses
                )

    # Fallback response
    return (
        "Sorry, I don't have an answer "
        "for that question yet. "
        "Please try asking another question."
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        model,
        tokenizer,
        label_encoder,
        intents
    ) = load_chatbot()

except Exception as e:

    st.error(
        "❌ Chatbot could not be loaded."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Student Chatbot")

st.write(
    "Ask questions about Artificial Intelligence, "
    "Machine Learning, Deep Learning, Python, NLP "
    "and other student topics."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 About")

    st.write(
        "This AI Student Chatbot uses "
        "Deep Learning and an LSTM model "
        "to understand student questions "
        "and predict the appropriate intent."
    )

    st.divider()

    st.write("### 🛠️ Technologies")

    st.write(
        """
        - Python
        - TensorFlow
        - Keras
        - LSTM
        - NLP
        - Streamlit
        """
    )

    st.divider()

    st.write(
        f"**Model Input Length:** {MAX_SEQUENCE_LENGTH}"
    )

    st.write(
        f"**Confidence Threshold:** "
        f"{CONFIDENCE_THRESHOLD}"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
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
# PROCESS USER QUESTION
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_input
        )

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:

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

        # ----------------------------------------------------
        # Generate Response
        # ----------------------------------------------------

        if confidence >= CONFIDENCE_THRESHOLD:

            response = get_response(
                predicted_intent,
                intents
            )

        else:

            response = (
                "I'm not completely sure about "
                "that question. Please try asking "
                "the question in another way."
            )

    except Exception as e:

        response = (
            "Sorry, something went wrong "
            "while processing your question."
        )

        st.error(
            f"Error: {e}"
        )

        predicted_intent = "Error"
        confidence = 0.0
        input_sequence = []


    # --------------------------------------------------------
    # Display Assistant Response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(
            response
        )

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

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

        st.write(
            input_sequence.tolist()
            if hasattr(
                input_sequence,
                "tolist"
            )
            else input_sequence
        )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader(
    "💡 Try asking"
)

col1, col2 = st.columns(2)

with col1:

    st.info(
        "What is Machine Learning?"
    )

with col2:

    st.info(
        "What is Deep Learning?"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python, TensorFlow, Keras, "
    "LSTM and Streamlit"
)
