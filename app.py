import os
import json
import pickle

import numpy as np
import streamlit as st
import keras

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Student Chatbot",
    page_icon="🤖",
    layout="centered"
)


# =========================================================
# FILE PATHS
# =========================================================

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


# =========================================================
# LOAD CHATBOT FILES
# =========================================================

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
            f"Label encoder not found: {LABEL_ENCODER_PATH}"
        )

    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(
            f"Intents file not found: {INTENTS_PATH}"
        )

    # Load Keras model
    model = keras.saving.load_model(
        MODEL_PATH,
        compile=False
    )

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    # Load label encoder
    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    # Load intents
    with open(INTENTS_PATH, "r", encoding="utf-8") as file:
        intents = json.load(file)

    return model, tokenizer, label_encoder, intents


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    model, tokenizer, label_encoder, intents = load_chatbot()

except Exception as e:

    st.error("❌ Chatbot could not be loaded.")

    st.code(str(e))

    st.info(
        "Please check chatbot_model.keras, tokenizer.pkl, "
        "label_encoder.pkl and data/intents.json."
    )

    st.stop()


# =========================================================
# GET RESPONSE FROM INTENTS.JSON
# =========================================================

def normalize_tag(tag):
    return (
        str(tag)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def get_response(intent_name):
    predicted_tag = normalize_tag(intent_name)

    for intent in intents.get("intents", []):
        json_tag = normalize_tag(intent.get("tag", ""))

        if json_tag == predicted_tag:
            responses = intent.get("responses", [])

            if responses:
                return random.choice(responses)

    return "Sorry, I don't have an answer for that question yet. Please try asking another question."
    )


# =========================================================
# PREDICT USER INTENT
# =========================================================

def predict_intent(user_text):

    # Convert user text into token numbers
    sequence = tokenizer.texts_to_sequences(
        [user_text]
    )

    # Model was trained with fixed sequence length = 6
    sequence = pad_sequences(
        sequence,
        maxlen=6,
        padding="post",
        truncating="post"
    )

    # Model prediction
    prediction = model.predict(
        sequence,
        verbose=0
    )

    # Highest probability index
    predicted_index = int(
        np.argmax(prediction[0])
    )

    # Confidence score
    confidence = float(
        np.max(prediction[0])
    )

    # Convert class index to intent name
    intent_name = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return intent_name, confidence


# =========================================================
# MAIN UI
# =========================================================

st.title("🤖 AI Student Chatbot")

st.write(
    "Ask questions about Artificial Intelligence, "
    "Machine Learning, Deep Learning, Python and Data Science."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Student Assistant")

    st.write(
        "This chatbot uses a Deep Learning LSTM model "
        "to understand the user's question and predict "
        "the correct intent."
    )

    st.divider()

    st.subheader("💡 Example Questions")

    st.write("• What is Machine Learning?")
    st.write("• What is Deep Learning?")
    st.write("• What is Python?")
    st.write("• What is Artificial Intelligence?")
    st.write("• What is Data Science?")

    st.divider()

    st.subheader("🧠 Model")

    st.write("Architecture: LSTM")
    st.write("Framework: TensorFlow / Keras")

    st.divider()

    st.caption("AI Student Chatbot")


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input(
    "Ask your question..."
)


# =========================================================
# PROCESS USER QUESTION
# =========================================================

if user_input:

    # Display user message
    with st.chat_message("user"):

        st.write(user_input)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:

        # Predict intent
        intent_name, confidence = predict_intent(
            user_input
        )

        # Get chatbot response
        response = get_response(
            intent_name
        )

        # Confidence threshold
        if confidence < 0.35:

            response = (
                "🤔 I'm not fully sure about that question.\n\n"
                "Please try asking in a different way."
            )

        # Display assistant response
        with st.chat_message("assistant"):

            st.write(response)

            # Debug information
            with st.expander(
                "🔎 Prediction Details"
            ):

                st.write(
                    "Predicted Intent:",
                    intent_name
                )

                st.write(
                    "Confidence:",
                    f"{confidence:.2%}"
                )

                st.write(
                    "Input Tokens:",
                    tokenizer.texts_to_sequences(
                        [user_input]
                    )
                )

    except Exception as e:

        with st.chat_message("assistant"):

            st.error(
                "❌ Sorry, something went wrong "
                "while processing your question."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(str(e))

        response = (
            "Sorry, I couldn't process that question."
        )

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Python, TensorFlow, Keras, LSTM and Streamlit"
)
