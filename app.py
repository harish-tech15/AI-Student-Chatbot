import os
import json
import pickle
import random

import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="AI Student Chatbot",
    page_icon="🤖",
    layout="centered"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
