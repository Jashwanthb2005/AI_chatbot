import os
import ssl
import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL fix before downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure nltk data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.path.append(os.path.abspath("nltk_data"))

# --- Training data ---
training_data = {
    "greeting": {
        "patterns": ["Hi", "Hello", "How are you?", "Is anyone there?", "Good day"],
        "response": ["Hello!", "Hi there!", "Greetings! How can I assist you?"]
    },
    "goodbye": {
        "patterns": ["Bye", "See you later", "Goodbye"],
        "response": ["Goodbye! Have a great day!", "See you later!"]
    },
    "thanks": {
        "patterns": ["Thanks", "Thank you", "That's helpful"],
        "response": ["You're welcome!", "Happy to help!"]
    },
    "help": {
        "patterns": ["Can you help me?", "I need assistance", "Help me"],
        "response": ["Sure! How can I assist you?", "I'm here to help! What do you need?"]
    },
    "name": {
        "patterns": ["What is your name?", "Who are you?", "Identify yourself"],
        "response": ["I am your friendly chatbot!", "You can call me Chatbot."]
    },
    "default": {
        "patterns": [],
        "response": ["I'm not sure I understand. Can you rephrase?", "Sorry, I didn't get that. Can you ask something else?"]
    }
}

# --- Prepare training data ---
patterns = []
labels = []

for intent, data in training_data.items():
    for pattern in data.get("patterns", []):
        patterns.append(pattern)
        labels.append(intent)

vectorizer = TfidfVectorizer(
    tokenizer=nltk.word_tokenize,
    token_pattern=None
)

X = vectorizer.fit_transform(patterns)

model = LogisticRegression()
model.fit(X, labels)

# --- Chatbot response ---
def chatbot_response(user_input):
    X_test = vectorizer.transform([user_input])
    intent = model.predict(X_test)[0]
    
    if intent in training_data:
        return random.choice(training_data[intent]["response"])
    else:
        return random.choice(training_data["default"]["response"])

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot", page_icon=":robot_face:")
st.title("Chatbot Application")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your message here...")

if user_input:
    bot_reply = chatbot_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_reply))

for sender, message in st.session_state.history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
