import os
import json
import pickle
import random

import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------

# PAGE CONFIG

# --------------------------------------------------

st.set_page_config(
page_title="AI Student Chatbot",
page_icon="🤖",
layout="centered"
)

# --------------------------------------------------

# BASE DIRECTORY

# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(**file**))

MODEL_PATH = os.path.join(
BASE_DIR, "model", "chatbot_model.keras"
)

TOKENIZER_PATH = os.path.join(
BASE_DIR, "model", "tokenizer.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
BASE_DIR, "model", "label_encoder.pkl"
)

INTENTS_PATH = os.path.join(
BASE_DIR, "data", "intents.json"
)

# --------------------------------------------------

# CHECK FILES

# --------------------------------------------------

required_files = {
"Model": MODEL_PATH,
"Tokenizer": TOKENIZER_PATH,
"Label Encoder": LABEL_ENCODER_PATH,
"Intents": INTENTS_PATH
}

missing_files = []

for name, path in required_files.items():
if not os.path.exists(path):
missing_files.append(f"{name}: {path}")

if missing_files:
st.error("❌ Required files are missing:")

```
for file in missing_files:
    st.write(file)

st.info(
    "Check your GitHub folder structure. "
    "Model files must be inside model/ and intents.json inside data/."
)

st.stop()
```

# --------------------------------------------------

# LOAD CHATBOT

# --------------------------------------------------

@st.cache_resource
def load_chatbot():

```
# Load trained model
model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Load intents JSON
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

return model, tokenizer, label_encoder, intents
```

# --------------------------------------------------

# LOAD FILES

# --------------------------------------------------

try:

```
model, tokenizer, label_encoder, intents = load_chatbot()
```

except Exception as e:

```
st.error("❌ Error loading chatbot files.")

st.code(str(e))

st.info(
    "Make sure your TensorFlow/Keras versions match "
    "the versions used during training."
)

st.stop()
```

# --------------------------------------------------

# NORMALIZE TAG

# --------------------------------------------------

def normalize_tag(tag):

```
return (
    str(tag)
    .strip()
    .lower()
    .replace("-", "_")
    .replace(" ", "_")
)
```

# --------------------------------------------------

# GET RESPONSE

# --------------------------------------------------

def get_response(intent_name, intents):

```
predicted_tag = normalize_tag(intent_name)

for intent in intents.get("intents", []):

    json_tag = normalize_tag(
        intent.get("tag", "")
    )

    if json_tag == predicted_tag:

        responses = intent.get(
            "responses",
            []
        )

        if responses:

            return random.choice(responses)

return (
    "Sorry, I don't have an answer for that "
    "question yet. Please try asking another question."
)
```

# --------------------------------------------------

# PREDICT INTENT

# --------------------------------------------------

def predict_intent(user_text):

```
# Convert text to sequence
sequence = tokenizer.texts_to_sequences(
    [user_text]
)

# Model input length
input_length = 6

# Pad sequence
sequence = pad_sequences(
    sequence,
    maxlen=input_length,
    padding="post",
    truncating="post"
)

# Prediction
prediction = model.predict(
    sequence,
    verbose=0
)

# Highest probability class
predicted_index = int(
    np.argmax(prediction[0])
)

confidence = float(
    np.max(prediction[0])
)

# Convert class index to intent name
intent_name = label_encoder.inverse_transform(
    [predicted_index]
)[0]

return (
    str(intent_name),
    confidence,
    sequence
)
```

# --------------------------------------------------

# TITLE

# --------------------------------------------------

st.title("🤖 AI Student Chatbot")

st.write(
"Ask questions about Machine Learning, "
"Deep Learning, Python, AI and Data Science."
)

# --------------------------------------------------

# SESSION STATE

# --------------------------------------------------

if "messages" not in st.session_state:

```
st.session_state.messages = []
```

# --------------------------------------------------

# DISPLAY CHAT HISTORY

# --------------------------------------------------

for message in st.session_state.messages:

```
with st.chat_message(
    message["role"]
):

    st.write(
        message["content"]
    )
```

# --------------------------------------------------

# CHAT INPUT

# --------------------------------------------------

user_input = st.chat_input(
"Ask your question..."
)

if user_input:

```
# Display user message
st.session_state.messages.append(
    {
        "role": "user",
        "content": user_input
    }
)

with st.chat_message("user"):

    st.write(user_input)


# Predict
try:

    intent_name, confidence, sequence = (
        predict_intent(user_input)
    )

    # Confidence threshold
    if confidence >= 0.35:

        response = get_response(
            intent_name,
            intents
        )

    else:

        response = (
            "I'm not very confident about "
            "that question. Please try "
            "rephrasing it."
        )

except Exception as e:

    intent_name = "prediction_error"

    confidence = 0.0

    sequence = []

    response = (
        "Sorry, I couldn't process that "
        "question."
    )

    st.error(str(e))


# Display assistant response
with st.chat_message("assistant"):

    st.write(response)


# Save assistant response
st.session_state.messages.append(
    {
        "role": "assistant",
        "content": response
    }
)


# --------------------------------------------------
# DEBUG INFORMATION
# --------------------------------------------------

with st.expander(
    "🔎 Prediction Details"
):

    st.write(
        "**Predicted Intent:**",
        intent_name
    )

    st.write(
        "**Confidence:**",
        f"{confidence * 100:.2f}%"
    )

    st.write(
        "**Input Tokens:**",
        sequence.tolist()
        if hasattr(sequence, "tolist")
        else sequence
    )
```

# --------------------------------------------------

# SIDEBAR

# --------------------------------------------------

with st.sidebar:

```
st.header("📚 About")

st.write(
    "This AI Student Chatbot is a "
    "Deep Learning based chatbot."
)

st.write("### Technologies")

st.write(
    """
    - Python
    - TensorFlow
    - Keras
    - NLP
    - Deep Learning
    - Streamlit
    """
)

st.write("### Example Questions")

st.write(
    """
    • What is Machine Learning?

    • What is Deep Learning?

    • What is Python?

    • What is NLP?

    • What is Generative AI?
    """
)
```
