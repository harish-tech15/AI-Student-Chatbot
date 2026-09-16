import os
import json
import pickle

import numpy as np
import streamlit as st
import keras

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

```
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
        f"Label encoder not found: {LABEL_ENCODER_PATH}"
    )

# Check intents
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

# Load intents JSON
with open(INTENTS_PATH, "r", encoding="utf-8") as file:
    intents = json.load(file)

return model, tokenizer, label_encoder, intents
```

# =========================================================

# LOAD EVERYTHING

# =========================================================

try:

```
model, tokenizer, label_encoder, intents = load_chatbot()
```

except Exception as e:

```
st.error("❌ Chatbot could not be loaded.")

st.code(str(e))

st.info(
    "Please check chatbot_model.keras, tokenizer.pkl, "
    "label_encoder.pkl and data/intents.json."
)

st.stop()
```

# =========================================================

# GET RESPONSE FROM INTENTS.JSON

# =========================================================

def get_response(intent_name):

```
if not isinstance(intents, dict):
    return "Sorry, I could not read the chatbot knowledge base."

intent_list = intents.get("intents", [])

for intent in intent_list:

    tag = intent.get("tag", "")

    if tag == intent_name:

        responses = intent.get("responses", [])

        if responses:

            return responses[0]

return (
    "Sorry, I don't have an answer for that question yet. "
    "Please try asking another question."
)
```

# =========================================================

# PREDICT USER INTENT

# =========================================================

def predict_intent(user_text):

```
# Convert text into numbers
sequence = tokenizer.texts_to_sequences(
    [user_text]
)

# Convert to NumPy array
sequence = np.array(sequence)

# Model prediction
prediction = model.predict(
    sequence,
    verbose=0
)

# Find highest probability
predicted_index = int(
    np.argmax(prediction[0])
)

confidence = float(
    np.max(prediction[0])
)

# Convert number back to intent name
intent_name = label_encoder.inverse_transform(
    [predicted_index]
)[0]

return intent_name, confidence
```

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

```
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
```

# =========================================================

# CHAT HISTORY

# =========================================================

if "messages" not in st.session_state:

```
st.session_state.messages = []
```

# Display previous messages

for message in st.session_state.messages:

```
with st.chat_message(
    message["role"]
):

    st.write(
        message["content"]
    )
```

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

```
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

    # Get response
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

        # Small debugging information
        with st.expander("🔎 Prediction Details"):

            st.write(
                "Predicted Intent:",
                intent_name
            )

            st.write(
                "Confidence:",
                f"{confidence:.2%}"
            )

except Exception as e:

    with st.chat_message("assistant"):

        st.error(
            "❌ Sorry, something went wrong "
            "while processing your question."
        )

        with st.expander("Technical Details"):

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
```

# =========================================================

# FOOTER

# =========================================================

st.divider()

st.caption(
"Built with Python, TensorFlow, Keras, LSTM and Streamlit"
)
