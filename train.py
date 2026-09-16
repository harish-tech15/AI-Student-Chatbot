import json
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Load Dataset
# -----------------------------

with open("data/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


patterns = []
labels = []


for intent in data["intents"]:

    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])


# -----------------------------
# Encode Labels
# -----------------------------

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)


# -----------------------------
# Tokenization
# -----------------------------

tokenizer = Tokenizer(
    num_words=1000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(patterns)

sequences = tokenizer.texts_to_sequences(patterns)


# -----------------------------
# Padding
# -----------------------------

max_length = 20

X = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="post"
)

y = np.array(encoded_labels)


# -----------------------------
# Build LSTM Model
# -----------------------------

model = Sequential([
    Embedding(
        input_dim=1000,
        output_dim=64,
        input_length=max_length
    ),

    LSTM(64),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dropout(0.2),

    Dense(num_classes, activation="softmax")
])


# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# Train
# -----------------------------

model.fit(
    X,
    y,
    epochs=100,
    batch_size=8,
    verbose=1
)


# -----------------------------
# Save Model
# -----------------------------

model.save("model/chatbot_model.keras")


# -----------------------------
# Save Tokenizer
# -----------------------------

with open("model/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)


# -----------------------------
# Save Label Encoder
# -----------------------------

with open("model/label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)


print("Training completed successfully!")
print("Model and preprocessing files saved.")