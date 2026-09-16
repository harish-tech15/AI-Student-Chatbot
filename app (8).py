import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Student Tutor",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# Load Model and Files
# -----------------------------
@st.cache_resource
def load_chatbot():

    model = load_model("model/chatbot_model.keras")

    with open("model/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open("model/label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)

    with open("data/intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)

    return model, tokenizer, label_encoder, intents


model, tokenizer, label_encoder, intents = load_chatbot()


# -----------------------------
# Find Response
# -----------------------------
def get_response(user_input):

    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=20, padding="post")

    prediction = model.predict(padded, verbose=0)

    confidence = float(np.max(prediction))
    predicted_index = int(np.argmax(prediction))

    predicted_tag = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Confidence check
    if confidence < 0.50:
        return (
            "I'm not fully sure about that question. "
            "Please ask me about Python, SQL, Machine Learning, "
            "Deep Learning, Data Science, CNN, or LSTM."
        )

    # Find matching intent
    for intent in intents["intents"]:

        if intent["tag"] == predicted_tag:

            responses = intent["responses"]

            # Simple deterministic response
            return responses[0]

    return "Sorry, I couldn't understand your question."


# -----------------------------
# UI
# -----------------------------
st.title("🎓 AI Student Tutor")
st.write(
    "Ask questions about Python, SQL, Data Science, "
    "Machine Learning and Deep Learning."
)


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Ask your question...")


if user_input:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Bot response
    response = get_response(user_input)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    with st.chat_message("assistant"):
        st.write(response)