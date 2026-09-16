# 🎓 AI Student Tutor Chatbot

An AI-powered student tutoring chatbot built using **Deep Learning and LSTM (Long Short-Term Memory)**.

The chatbot understands student questions related to Python, SQL, Data Science, Machine Learning, Deep Learning, CNN, and LSTM, and returns an appropriate response based on the predicted intent.

## 🚀 Project Overview

Students often need quick explanations while learning technical subjects.

This project uses an **LSTM neural network** to classify a user's question into a predefined intent and provide a relevant educational response.

### Workflow

User Question
↓
Text Tokenization
↓
Sequence Padding
↓
LSTM Deep Learning Model
↓
Intent Classification
↓
Response Selection
↓
Chatbot Response

## 🧠 Technologies Used

* Python
* TensorFlow
* Keras
* LSTM
* NumPy
* Pandas
* Scikit-learn
* NLTK
* JSON
* Streamlit

## ✨ Features

* 🎓 Student-focused AI tutor
* 💬 Interactive chatbot interface
* 🧠 LSTM-based intent classification
* 🐍 Python learning questions
* 🗄️ SQL learning questions
* 📊 Data Science questions
* 🤖 Machine Learning questions
* 🧠 Deep Learning questions
* 🖼️ CNN questions
* 🔄 LSTM questions
* 📱 Streamlit web interface
* ⚡ Confidence-based fallback response

## 📂 Project Structure

```text
AI-Student-Chatbot/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── intents.json
│
├── model/
│   ├── chatbot_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
└── screenshots/
    └── chatbot.png
```

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd AI-Student-Chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Example Questions

Try asking:

* What is Python?
* What is Machine Learning?
* What is Deep Learning?
* What is CNN?
* What is LSTM?
* What is SQL?
* What is Data Science?

## 🏗️ Model Architecture

The chatbot uses the following neural network architecture:

```text
Input Text
    ↓
Tokenizer
    ↓
Sequence Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dropout
    ↓
Dense Layer
    ↓
Dropout
    ↓
Softmax Output
```

## 📌 Model Details

* Embedding dimension: 64
* LSTM units: 64
* Dense layer: 32 neurons
* Output activation: Softmax
* Optimizer: Adam
* Loss function: Sparse Categorical Crossentropy
* Training epochs: 100
* Batch size: 8

## 🎯 Learning Outcomes

Through this project, I practiced:

* Natural Language Processing fundamentals
* Text tokenization
* Sequence padding
* Label encoding
* Deep Learning
* LSTM networks
* Intent classification
* Model saving and loading
* Streamlit application development
* GitHub project organization

## ⚠️ Limitations

This is an intent-classification chatbot rather than a general-purpose conversational AI system.

Its responses are limited to the intents and training examples contained in the dataset.

A future version could use Transformer models, RAG, larger datasets, conversation memory, and a more advanced NLP architecture.

## 🔮 Future Improvements

* Add more student topics
* Expand the training dataset
* Add Transformer-based NLP
* Add RAG for educational documents
* Add quiz generation
* Add student progress tracking
* Add conversation history
* Add voice input
* Deploy the application online

## 👨‍💻 Author

**Harish S.**

B.Tech Artificial Intelligence and Data Science

## ⭐ Project Purpose

This project was developed as a practical Deep Learning portfolio project demonstrating how an LSTM model can be integrated into a real-world educational chatbot application.

